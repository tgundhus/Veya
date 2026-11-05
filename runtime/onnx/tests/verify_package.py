"""Verify base inference and optional services in an isolated Python environment."""
import argparse
from email.parser import BytesParser
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import venv
import zipfile


def command(*args, cwd, env):
    subprocess.run([str(arg) for arg in args], cwd=cwd, env=env, check=True)


def check_wheel(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        metadata = [name for name in names if name.endswith(".dist-info/METADATA")]
        assert len(metadata) == 1
        message = BytesParser().parsebytes(archive.read(metadata[0]))
        assert message["Name"] == "veya-onnx"
        requirements = message.get_all("Requires-Dist") or []
        assert all(not any(req.lower().startswith(name) for name in (
            "torch", "transformers", "safetensors", "onnxscript", "veya ")) for req in requirements), requirements
        assert {req.split(">=")[0] for req in requirements if ";" not in req} == {
            "numpy", "onnxruntime", "tokenizers", "huggingface_hub"}, requirements
        assert set(message.get_all("Provides-Extra")) == {"structured", "serve", "mcp"}
        for dependency, extra in (("fastapi", "serve"), ("uvicorn", "serve"), ("mcp", "mcp")):
            assert any(req.startswith(dependency + ">=") and ('extra == "' + extra + '"') in req
                       for req in requirements), requirements
        entry_points, = [name for name in names if name.endswith(".dist-info/entry_points.txt")]
        scripts = archive.read(entry_points).decode()
        assert "laya-onnx-serve = laya_onnx.serve:main" in scripts, scripts
        assert "laya-onnx-mcp = laya_onnx.mcp.server:main" in scripts, scripts
        assert "laya_onnx/py.typed" in names
        for module in ("router", "serve", "shortlist", "lang", "presets", "mcp/server", "mcp/tools"):
            assert "laya_onnx/_runtime/" + module + ".py" in names, module
        assert any("LICENSE" in name for name in names)
        assert all(name.startswith(("laya_onnx/", "veya_onnx-")) for name in names), names
        assert not any(name.endswith(("/agent.py", "/common.py", "/calibrate.py")) for name in names)


def extract_sdist(source, destination):
    if not hasattr(tarfile, "data_filter"):
        raise RuntimeError("Package verification needs a Python with the tarfile security backport")
    destination = destination.resolve()
    with tarfile.open(source) as archive:
        for entry in archive.getmembers():
            target = (destination / entry.name).resolve()
            if Path(entry.name).is_absolute() or not target.is_relative_to(destination):
                raise ValueError("sdist member escapes its extraction directory")
            if not (entry.isfile() or entry.isdir()):
                raise ValueError("sdist must contain regular files and directories only")
        archive.extractall(destination, filter="data")


def check_service_commands(python, root, env):
    scripts = python.parent
    for name in ("laya-onnx-serve", "laya-onnx-mcp"):
        executable = scripts / (name + ".exe" if os.name == "nt" else name)
        assert executable.is_file(), executable
        output = subprocess.check_output([str(executable), "--help"], cwd=root, env=env, text=True)
        assert "--backend {torch,onnx}" in output, output
        assert "--performance-mode" in output, output
    for module in ("laya_onnx.serve", "laya_onnx.mcp.server"):
        output = subprocess.check_output([str(python), "-I", "-m", module, "--help"],
                                         cwd=root, env=env, text=True)
        assert "--backend {torch,onnx}" in output, output


def distributions(python, root, env):
    code = ("import importlib.metadata as m,json; print(json.dumps({d.metadata['Name']: "
            "{'version':d.version,'bytes':sum(p.locate().stat().st_size for p in d.files or [] "
            "if p.locate().is_file())} for d in m.distributions()},indent=2))")
    return json.loads(subprocess.check_output([str(python), "-I", "-c", code], cwd=root, env=env, text=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dist", type=Path)
    parser.add_argument("--environment", type=Path, help="keep the isolated environment here for benchmarks")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    tests = Path(__file__).resolve().parent
    wheels, sdists = list(args.dist.resolve().glob("*.whl")), list(args.dist.resolve().glob("*.tar.gz"))
    assert len(wheels) == len(sdists) == 1, "Use a clean output directory with one wheel and sdist"
    check_wheel(wheels[0])
    env = {key: value for key, value in os.environ.items() if not key.startswith("LAYA_")}
    env.update(PIP_DISABLE_PIP_VERSION_CHECK="1", TOKENIZERS_PARALLELISM="false")
    env.pop("PYTHONPATH", None)
    with tempfile.TemporaryDirectory(prefix="laya-onnx-package-") as tmp:
        root = Path(tmp)
        fixture = root / "fixture"
        command(sys.executable, tests / "fixture.py", fixture, cwd=root, env=env)
        environment = args.environment.resolve() if args.environment else root / "environment"
        if environment.exists():
            raise ValueError("verification requires a new isolated environment: %s" % environment)
        venv.EnvBuilder(with_pip=True).create(environment)
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        command(python, "-m", "pip", "install", wheels[0], cwd=root, env=env)
        command(python, "-I", tests / "consumer.py", fixture, cwd=root, env=env)
        base_distributions = distributions(python, root, env)
        assert not {"fastapi", "uvicorn", "mcp", "starlette", "pydantic"} & {
            name.lower() for name in base_distributions}, base_distributions
        command(python, "-m", "pip", "install", str(wheels[0]) + "[serve,mcp]", cwd=root, env=env)
        check_service_commands(python, root, env)
        command(python, "-I", tests / "services_consumer.py", fixture, cwd=root, env=env)
        unpacked = root / "source"
        unpacked.mkdir()
        extract_sdist(sdists[0], unpacked)
        projects = list(unpacked.iterdir())
        assert len(projects) == 1
        for module in ("onnx_agent", "serve", "router", "mcp/server", "mcp/tools"):
            assert (projects[0] / "_shared" / (module + ".py")).is_file(), module
        rebuilt = root / "rebuilt"
        command(sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", rebuilt,
                projects[0], cwd=root, env=env)
        rebuilt_wheel, = rebuilt.glob("*.whl")
        check_wheel(rebuilt_wheel)
        command(python, "-m", "pip", "install", "--no-deps", "--force-reinstall", rebuilt_wheel, cwd=root, env=env)
        command(python, "-I", tests / "consumer.py", fixture, cwd=root, env=env)
        check_service_commands(python, root, env)
        command(python, "-I", tests / "services_consumer.py", fixture, cwd=root, env=env)
        command(python, "-m", "pip", "check", cwd=root, env=env)
        if args.report:
            report = distributions(python, root, env)
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps({"distributions": report, "base_distributions": base_distributions,
                                               "wheel_bytes": wheels[0].stat().st_size,
                                               "sdist_bytes": sdists[0].stat().st_size}, indent=2) + "\n", encoding="utf-8")
    print("PASS isolated base/HTTP/MCP wheel, outside-checkout sdist rebuild and real ONNX Runtime contracts")


if __name__ == "__main__":
    main()
