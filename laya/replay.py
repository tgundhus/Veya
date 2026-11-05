"""Verified, opt-in exact cache replay without constructing inference models.

The manifest is trusted deployment metadata, not an executable object or a remote lookup.
It freezes the inputs needed by the existing decision-cache key. Arbitrary hooks and
subclasses retain the ordinary prediction path.
"""
import copy
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from contextlib import nullcontext
from types import SimpleNamespace

from .errors import DecisionCacheError

_SCHEMA = "veya-replay-v1"
_MAX_MANIFEST_BYTES = 8 * 1024 * 1024


def _versions():
    from . import __version__, lang
    from .consistency import _KEY_VERSION
    from ._routing_unicode import TABLE_SHA256
    return {"library": __version__, "key": _KEY_VERSION,
            "routing": getattr(lang, "ROUTING_CONTRACT_VERSION", "legacy"),
            "unicode_tables": TABLE_SHA256}


def _encode(value):
    from .consistency import _typed_json
    return json.dumps(_typed_json(value), ensure_ascii=True, separators=(",", ":"),
                      allow_nan=False).encode("ascii")


def load_manifest(path):
    """Read bounded non-executable metadata and reject corruption or incompatible contracts."""
    from .consistency import _from_typed_json
    try:
        with open(path, "rb") as source:
            raw = source.read(_MAX_MANIFEST_BYTES + 1)
        if len(raw) > _MAX_MANIFEST_BYTES:
            raise ValueError("manifest exceeds the size limit")
        envelope = json.loads(raw)
        if not isinstance(envelope, dict) or set(envelope) != {"schema", "sha256", "payload"}:
            raise ValueError("invalid manifest envelope")
        if envelope["schema"] != _SCHEMA:
            raise ValueError("unsupported manifest schema")
        payload = _from_typed_json(envelope["payload"])
        digest = hashlib.sha256(_encode(payload)).hexdigest()
        if not isinstance(envelope["sha256"], str) or not hmac.compare_digest(digest, envelope["sha256"]):
            raise ValueError("manifest checksum does not match")
        if (not isinstance(payload, dict) or payload.get("versions") != _versions()
                or not isinstance(payload.get("models"), dict) or not payload["models"]
                or not isinstance(payload.get("store_id"), str)
                or not isinstance(payload.get("configuration"), dict)):
            raise ValueError("manifest contract does not match this runtime")
        return payload
    except Exception as exc:
        raise DecisionCacheError("replay manifest is unavailable, corrupt or incompatible") from exc


def _configuration(router):
    from .router import _split
    sources = {}
    for name, spec in router.models.items():
        if spec is None:
            sources[name] = None
            continue
        source, subfolder = _split(spec)
        if os.path.isdir(source) or os.path.isabs(source) or source.startswith(("./", "../")):
            sources[name] = {"kind": "local", "directory": str(Path(source, subfolder).resolve()
                                                                if subfolder else Path(source).resolve()),
                             "source": source, "subfolder": subfolder}
        else:
            sources[name] = {"kind": "hub", "source": source, "subfolder": subfolder}
    arguments = copy.deepcopy(router.agent_kwargs)
    if arguments.get("calibration") is not None:
        arguments["calibration"] = str(Path(arguments["calibration"]).resolve())
    # Runtime precision/device and memory budgets do not change final-decision identity.
    # Model and calibration construction inputs do: changing them requires re-export.
    return {"models": sources, "default": router.default,
            "auto_task_detection": router.auto_task_detection, "lang_guess": router.lang_guess,
            "revision": router.revision, "revisions": copy.deepcopy(router.revisions),
            "agent_kwargs": arguments,
            "sha256_digests": copy.deepcopy(router.sha256_digests),
            "backend": router.backend, "onnx_paths": copy.deepcopy(router.onnx_paths),
            "revision_environment": (os.environ.get("LAYA_REVISION") or "").strip() or None,
            "digest_environment": (os.environ.get("LAYA_SHA256_DIGESTS") or "").strip() or None}


def validate_constructor_options(metadata, arguments):
    """Use administrator-approved options without importing Agent just to read its signature."""
    from .router import _ROUTER_OWNED_AGENT_ARGS
    reserved = sorted(set(arguments) & _ROUTER_OWNED_AGENT_ARGS)
    if reserved:
        raise ValueError("Router owns these agent options: %s" % ", ".join(reserved))
    normalized = copy.deepcopy(arguments)
    if normalized.get("calibration") is not None:
        normalized["calibration"] = str(Path(normalized["calibration"]).resolve())
    try:
        if _encode(normalized) != _encode(metadata["configuration"]["agent_kwargs"]):
            raise DecisionCacheError("agent options differ from the approved replay configuration")
    except DecisionCacheError:
        raise
    except Exception as exc:
        raise DecisionCacheError("approved replay agent options could not be verified") from exc


def _builtin_agent(agent):
    types = []
    for module, names in (("agent", ("Agent",)),
                          ("onnx_agent", ("ONNXAgent", "_NativeONNXAgent"))):
        loaded = sys.modules.get(__package__ + "." + module)
        types.extend(getattr(loaded, name, None) for name in names)
    return type(agent) in types


def _cache(router, active=None):
    from .consistency import DecisionCache, _SQLiteStore
    from .hooks import compose_hooks
    from .router import Router
    if type(router) is not Router or callable(router.lang_guess):
        return None
    hooks = compose_hooks(router.hooks) if active is None else active
    caches = [hook for hook in hooks if type(hook) is DecisionCache]
    if not caches and getattr(router, "replay_manifest", None) is not None:
        raise DecisionCacheError("configured replay metadata requires a built-in SQLite decision cache")
    if len(caches) != 1:
        return None
    if type(caches[0]._store) is not _SQLiteStore:
        if getattr(router, "replay_manifest", None) is not None:
            raise DecisionCacheError("configured replay metadata requires a built-in SQLite decision cache")
        return None
    if len(hooks) != 1:
        return None
    # A retention callback is arbitrary hook code. Let normal dispatch preserve its
    # suppression/deadline/serialization policy instead of invoking it speculatively.
    if callable(caches[0].ttl):
        return None
    if router.hooks_timeout is not None:
        return None
    return hooks[0]


def _stat(path):
    item = os.stat(path)
    return [item.st_size, item.st_mtime_ns, item.st_ino]


def _onnx_artifacts(path):
    """Use the ONNX schema to approve graph weights, including nested/sparse tensors."""
    try:
        import onnx
    except ImportError as exc:
        raise ImportError("preparing an ONNX replay manifest requires the optional replay extra (onnx>=1.20)") from exc
    # ORT resolves sidecars relative to the supplied graph's parent, including when the
    # graph itself is a symlink. Keep logical file paths so retargeting is detected too.
    graph = Path(os.path.abspath(path))
    base = graph.parent.resolve()
    model = onnx.load(str(graph), load_external_data=False)
    files = {str(graph): _stat(graph)}
    pending = [model]
    while pending:
        message = pending.pop()
        if message.DESCRIPTOR.full_name == "onnx.TensorProto" and message.external_data:
            entries = {entry.key: entry.value for entry in message.external_data}
            location = entries.get("location", "")
            relative = Path(location)
            if (not location or relative.is_absolute() or relative.drive or ".." in relative.parts
                    or ":" in location or "\\" in location):
                raise ValueError("ONNX external weights must use safe relative paths within the graph directory")
            artifact = graph.parent / relative
            if not artifact.resolve().is_relative_to(base) or not artifact.is_file():
                raise ValueError("ONNX external weights must remain within the graph directory")
            files[str(artifact)] = _stat(artifact)
        # Reflection follows the installed official schema, including tensors in sparse
        # initializers, function attributes and arbitrarily nested graph attributes.
        for descriptor, value in message.ListFields():
            if descriptor.message_type is not None:
                pending.extend(value if descriptor.is_repeated else [value])
    return files


def _model_record(router, name, agent):
    from .consistency import _decision_policy, _fingerprint
    from .router import _split, _revision_pin
    if not _builtin_agent(agent) or (getattr(agent, "hooks", ()) or ()):
        raise ValueError("replay export requires built-in agents without prediction hooks")
    if router.models.get(name) is None:
        raise ValueError("replay export requires a registered rebuildable checkpoint source")
    record = {"fingerprint": copy.deepcopy(_fingerprint(agent)), "attributes": {}, "files": {}}
    for field in ("model_id", "revision", "subfolder", "_checkpoint_signature", "temperature",
                  "temperature_by_options", "lang_temperatures", "binning_map"):
        record["attributes"][field] = copy.deepcopy(getattr(agent, field, None))
    policy = _decision_policy(agent)
    if policy != "canonical":
        record["attributes"]["decision_policy"] = policy
    record["attributes"]["cfg"] = {key: copy.deepcopy(agent.cfg.get(key))
                                    for key in ("max_len", "head_max_len", "option_layout", "act_costs")}
    source, subfolder = _split(router.models[name])
    if os.path.isdir(source):
        directory = Path(source, subfolder) if subfolder else Path(source)
        signature = record["attributes"]["_checkpoint_signature"]
        if not isinstance(signature, list) or not signature:
            raise ValueError("local replay export requires load-time artifact identity")
        for entry in signature:
            filename = entry[0]
            path = directory / filename
            if filename not in ("rl_agent_config.json", "encoder/config.json") and not filename.startswith("tokenizer/"):
                if type(agent).__name__ == "_NativeONNXAgent" or filename != "model.safetensors":
                    path = Path(router.onnx_paths.get(name, str(path)))
            stamp = _stat(path)
            if entry[1] == "stat" and stamp != entry[2:]:
                raise ValueError("checkpoint files changed since the agent loaded; reload before exporting")
            if entry[1] == "sha256":
                digest = hashlib.sha256()
                with open(path, "rb") as source_file:
                    for block in iter(lambda: source_file.read(1024 * 1024), b""):
                        digest.update(block)
                if digest.hexdigest() != entry[2]:
                    raise ValueError("checkpoint artifact no longer matches its verified digest")
            record["files"][str(path.absolute())] = stamp
    else:
        revision = (_revision_pin(router.revisions.get(name)) or _revision_pin(router.revision)
                    or _revision_pin(os.environ.get("LAYA_REVISION") or None))
        if not isinstance(revision, str) or not re.fullmatch("[a-fA-F0-9]{40}", revision):
            raise ValueError("remote replay export requires an immutable 40-character checkpoint revision")
        if revision.lower() != str(getattr(agent, "revision", "")).lower():
            raise ValueError("resident checkpoint revision differs from the configured immutable revision")
    calibration = router.agent_kwargs.get("calibration")
    if calibration:
        record["files"][str(Path(calibration).resolve())] = _stat(calibration)
    if type(agent).__name__ in ("ONNXAgent", "_NativeONNXAgent"):
        graph = router.onnx_paths.get(name)
        if graph is None:
            raise ValueError("ONNX replay export requires its registered graph path in onnx_paths")
        record["files"].update(_onnx_artifacts(graph))
    # Validate the whole record is finite plain data, preserving integer map keys and tuples.
    _encode(record)
    return record


def export_manifest(router, path):
    """Capture resident model metadata and atomically publish a deployment manifest."""
    from .consistency import _canonical
    cache = _cache(router)
    if cache is None or cache.store_id is None:
        raise ValueError("replay export requires exactly one built-in SQLite DecisionCache and no custom hooks or deadlines")
    if router.agent_kwargs:
        from .router import check_agent_kwargs
        check_agent_kwargs(router.agent_kwargs, backend=router.backend)
    with router._lock:
        if not router._agents:
            raise ValueError("load the checkpoints to approve before exporting replay metadata")
        configuration = _configuration(router)
        body = {"versions": _versions(), "store_id": cache.store_id, "fingerprint": cache.fingerprint,
                "configuration": configuration,
                "models": {name: _model_record(router, name, agent) for name, agent in router._agents.items()}}
        encoded = _encode(body)
        if _canonical(configuration) != _canonical(_configuration(router)):
            raise ValueError("Router configuration changed during replay export")
    from .consistency import _typed_json
    envelope = {"schema": _SCHEMA, "sha256": hashlib.sha256(encoded).hexdigest(),
                "payload": _typed_json(body)}
    raw = json.dumps(envelope, ensure_ascii=True, indent=2, allow_nan=False).encode("ascii") + b"\n"
    if len(raw) > _MAX_MANIFEST_BYTES:
        raise ValueError("replay manifest exceeds the size limit")
    destination = Path(path).resolve()
    descriptor, temporary = tempfile.mkstemp(dir=destination.parent, prefix=".veya-replay-", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(raw)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    with router._lock:
        router._replay_metadata = copy.deepcopy(body)
        router._replay_path = str(destination)
    return {"path": str(destination), "store_id": cache.store_id,
            "models": list(body["models"]), "sha256": envelope["sha256"]}


def validate_loaded_agent(router, name, agent, active):
    """A cold miss must rebuild the approved model before it may commit a new decision."""
    from .consistency import _canonical, _fingerprint
    if getattr(router, "replay_manifest", None) is None:
        return
    cache = _cache(router, active)
    if cache is None:
        return
    try:
        with router._lock:
            metadata = router._replay_metadata
            if (metadata["versions"] != _versions() or metadata["store_id"] != cache.store_id
                    or metadata["fingerprint"] != cache.fingerprint
                    or _canonical(metadata["configuration"]) != _canonical(_configuration(router))):
                raise DecisionCacheError("replay manifest does not match this store or policy configuration")
            record = metadata["models"].get(name)
            if record is None:
                return
            if (not _builtin_agent(agent) or (getattr(agent, "hooks", ()) or ())
                    or _canonical(_fingerprint(agent)) != _canonical(record["fingerprint"])):
                raise DecisionCacheError("loaded checkpoint does not match its approved replay metadata")
            for filename, expected in record["files"].items():
                if _stat(filename) != expected:
                    raise DecisionCacheError("checkpoint files changed after replay approval")
    except DecisionCacheError:
        raise
    except Exception as exc:
        raise DecisionCacheError("loaded checkpoint could not be verified against replay metadata") from exc


def validate_manifest(router):
    """Check every approved record without reading decisions or importing inference runtimes."""
    from .consistency import _canonical, _fingerprint
    if getattr(router, "replay_manifest", None) is None:
        return False
    cache = _cache(router)
    if cache is None:
        return False
    try:
        with router._lock:
            metadata = router._replay_metadata
            if (metadata["versions"] != _versions() or metadata["store_id"] != cache.store_id
                    or metadata["fingerprint"] != cache.fingerprint
                    or _canonical(metadata["configuration"]) != _canonical(_configuration(router))):
                raise DecisionCacheError("replay manifest does not match this store or policy configuration")
            for name, record in metadata["models"].items():
                if name not in router.models:
                    raise DecisionCacheError("approved checkpoint is no longer registered")
                proxy = SimpleNamespace(**copy.deepcopy(record["attributes"]))
                if _canonical(_fingerprint(proxy)) != _canonical(record["fingerprint"]):
                    raise DecisionCacheError("replay manifest contains inconsistent checkpoint metadata")
                resident = router._agents.get(name)
                if resident is not None and (not _builtin_agent(resident)
                        or (getattr(resident, "hooks", ()) or ())
                        or _canonical(_fingerprint(resident)) != _canonical(record["fingerprint"])):
                    raise DecisionCacheError("resident checkpoint metadata changed after replay approval")
                for filename, expected in record["files"].items():
                    if _stat(filename) != expected:
                        raise DecisionCacheError("checkpoint files changed after replay approval")
        return True
    except DecisionCacheError:
        raise
    except Exception as exc:
        raise DecisionCacheError("replay deployment metadata could not be verified") from exc


def try_replay(router, state, questions, *, model=None, task=None, lang=None, lang_guess=None,
               max_len=None, head_max_len=None, min_confidence=None, active=None, decision=None):
    """Return an authoritative exact hit, or None without claiming or computing a miss."""
    from .confidence import apply_confidence_gate, check_min_confidence
    from .consistency import _canonical, _fingerprint
    from .hooks import PredictContext
    from .router import _record_calibration_language
    if state is None:
        raise TypeError("state must not be None; pass a string, dict, or list")
    if not isinstance(questions, dict):
        raise TypeError("questions must be a dict of question id -> definition")
    threshold = check_min_confidence(min_confidence) if min_confidence is not None else None
    if getattr(router, "replay_manifest", None) is None:
        return None
    cache = _cache(router, active)
    if cache is None or callable(lang_guess):
        return None
    # User routing errors retain their normal ValueError/TypeError contract (HTTP 422).
    # Only approval/storage failures belong to the authorization error boundary below.
    route = decision if decision is not None else router._route(
        state, questions, model=model, task=task, lang=lang, lang_guess=lang_guess)
    name = router.resolve(route["model"])
    try:
        with router._lock:
            metadata = router._replay_metadata
            if (metadata["versions"] != _versions()
                    or metadata["store_id"] != cache.store_id or metadata["fingerprint"] != cache.fingerprint
                    or _canonical(metadata["configuration"]) != _canonical(_configuration(router))):
                raise DecisionCacheError("replay manifest does not match this store or policy configuration")
            record = metadata["models"].get(name)
            if record is None:
                return None
            resident = router._agents.get(name)
            if resident is not None:
                if not _builtin_agent(resident) or (getattr(resident, "hooks", ()) or ()):
                    raise DecisionCacheError("approved checkpoint was replaced or given custom Agent hooks")
                if _canonical(_fingerprint(resident)) != _canonical(record["fingerprint"]):
                    raise DecisionCacheError("resident checkpoint metadata changed after replay approval")
            else:
                for filename, expected in record["files"].items():
                    if _stat(filename) != expected:
                        raise DecisionCacheError("checkpoint files changed after replay approval")
            proxy = SimpleNamespace(**copy.deepcopy(record["attributes"]))
            if _canonical(_fingerprint(proxy)) != _canonical(record["fingerprint"]):
                raise DecisionCacheError("replay manifest contains inconsistent checkpoint metadata")
            context = PredictContext(states=[state], questions=questions, decision=dict(route),
                                     model=name, agent=proxy, router=router, max_len=max_len, head_max_len=head_max_len)
            _record_calibration_language(context, lang)
            key, = cache._keys(context, context.states, bool(proxy.lang_temperatures),
                               fingerprint=metadata["fingerprint"])
        # Respect an explicitly serialized hook policy, without holding the metadata
        # lock while doing store I/O. The captured key cannot change namespaces.
        with getattr(router, "_hooks_lock", None) or nullcontext():
            replay = cache._lookup_replay(context, key=key)
        if replay is not None:
            replay["routing"] = dict(route)
            apply_confidence_gate([replay], threshold)
        return replay
    except DecisionCacheError:
        raise
    except Exception as exc:
        raise DecisionCacheError("verified decision replay could not be authorized") from exc


def main(argv=None):
    """Prepare a manifest using the same environment configuration as local HTTP/MCP."""
    import argparse
    from .serve import build_router, _close_decision_cache
    parser = argparse.ArgumentParser(description="Approve loaded local checkpoint metadata for verified decision replay")
    parser.add_argument("--output", required=True, help="local JSON manifest destination")
    parser.add_argument("--models", required=True, help="comma-separated registered model names to approve")
    args = parser.parse_args(argv)
    names = [name.strip() for name in args.models.split(",") if name.strip()]
    if not names:
        parser.error("--models must name at least one checkpoint")
    previous = {key: os.environ.get(key) for key in ("LAYA_PRELOAD", "LAYA_REPLAY_MANIFEST")}
    router = None
    try:
        # Preparation is explicit reapproval: an older/stale manifest must not prevent it.
        # Existing-store and expected-identity guards are deliberately retained.
        os.environ["LAYA_PRELOAD"] = "0"
        os.environ.pop("LAYA_REPLAY_MANIFEST", None)
        router = build_router()
        selected = list(dict.fromkeys(router.resolve(name) for name in names))
        if len(selected) > router.max_loaded:
            raise ValueError("preparing these checkpoints requires LAYA_MAX_LOADED >= %d" % len(selected))
        router.preload(selected)
        print(json.dumps(router.export_replay_manifest(args.output), sort_keys=True))
        return 0
    finally:
        try:
            if router is not None:
                try:
                    router.unload()
                finally:
                    _close_decision_cache(router)
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


if __name__ == "__main__":
    raise SystemExit(main())
