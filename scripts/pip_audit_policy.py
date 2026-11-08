"""Decide whether pip-audit's reports pass: the dependency audit's policy, in one place.

pip-audit fails on every advisory it finds. A few have no fixed release anywhere and cannot be
reached through Veya -- a hole in a server this package never starts, in a package only an
optional integration pulls in -- and a check that is red for those is red for everything, so it
stops being read. This script reads pip-audit's JSON reports and fails on:

- an advisory that is not in the accepted list,
- an accepted advisory once a fixed release exists: upgrade, or review the entry again,
- an entry accepted for one package that shows up on another, and
- a dependency pip-audit could not audit at all (what its --strict fails on).

An accepted advisory that is still unfixed passes with a warning, so it stays in sight. A report
passed with `--shipped` covers what the package and its image ship (the core plus `serve`): no
advisory there is ever accepted, whatever the list says. The list
lives in `.github/accepted-advisories.toml`; each entry needs an `id`, the `package` and a
`reason`, and may give `aliases` (CVE and GHSA ids) and where it comes from.

    python scripts/pip_audit_policy.py --accepted .github/accepted-advisories.toml \
        --shipped shipped.json report.json ...
"""
import argparse
import json
import sys
from typing import Any, Dict, List, Tuple


def _package(name: str) -> str:
    return name.lower().replace("_", "-")


def load_accepted(path: str) -> Dict[str, Dict[str, Any]]:
    """Accepted entries by every id they carry (the advisory id and its aliases)."""
    import tomllib  # Python 3.11+, what the audit job runs; the rest of this module imports on 3.10

    with open(path, "rb") as fh:
        entries = tomllib.load(fh).get("advisory", [])
    accepted: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        missing = [key for key in ("id", "package", "reason") if not entry.get(key)]
        if missing:
            raise ValueError("accepted advisory %r has no %s" % (entry.get("id"), ", ".join(missing)))
        for advisory_id in [entry["id"], *entry.get("aliases", [])]:
            accepted[advisory_id] = entry
    return accepted


def evaluate(reports: List[Dict[str, Any]], accepted: Dict[str, Dict[str, Any]]
             ) -> Tuple[List[str], List[str], List[str]]:
    """(failures, warnings, stale) for pip-audit JSON reports against the accepted entries.

    `stale` names accepted entries no report mentioned, which can go from the list.
    """
    failures: List[str] = []
    warnings: List[str] = []
    used = set()
    seen = set()
    for report in reports:
        for dep in report.get("dependencies", []):
            name, version = dep.get("name", "?"), dep.get("version", "?")
            if dep.get("skip_reason"):
                failures.append("%s: not audited: %s" % (name, dep["skip_reason"]))
                continue
            for vuln in dep.get("vulns", []):
                key = (_package(name), version, vuln["id"])
                if key in seen:                 # the same advisory, reported twice or by two groups
                    continue
                seen.add(key)
                ids = [vuln["id"], *vuln.get("aliases", [])]
                entry = next((accepted[i] for i in ids if i in accepted), None)
                fixes = vuln.get("fix_versions") or []
                label = "%s %s: %s" % (name, version, " / ".join(ids))
                if entry is None:
                    failures.append("%s is not accepted%s" % (
                        label, "; fixed in %s" % ", ".join(fixes) if fixes else ", and has no fix"))
                    continue
                used.add(entry["id"])
                if _package(entry["package"]) != _package(name):
                    failures.append("%s is accepted for %s, not %s" % (label, entry["package"], name))
                elif fixes:
                    failures.append("%s was accepted as unfixed and is now fixed in %s: upgrade, "
                                    "or review the entry again" % (label, ", ".join(fixes)))
                else:
                    warnings.append("%s: accepted, no fix released. %s" % (label, entry["reason"]))
    stale = sorted({entry["id"] for entry in accepted.values()} - used)
    return failures, warnings, stale


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--accepted", required=True, help="the accepted-advisories TOML file")
    parser.add_argument("--shipped", action="append", default=[], metavar="REPORT",
                        help="a report of what the package and its image ship; nothing in it is "
                             "accepted (repeatable)")
    parser.add_argument("reports", nargs="+", help="pip-audit reports written with -f json")
    args = parser.parse_args(argv)
    accepted = load_accepted(args.accepted)

    def read(paths):
        loaded = []
        for path in paths:
            with open(path, encoding="utf-8") as fh:
                loaded.append(json.load(fh))
        return loaded

    reports, shipped = read(args.reports), read(args.shipped)
    failures, warnings, stale = evaluate(reports, accepted)
    # The shipped scope gets no exceptions: an advisory there fails even if the list accepts it.
    failures += ["%s (it ships in the package or its image, where nothing is accepted)" % line
                 for line in evaluate(shipped, {})[0]]
    reports = reports + shipped
    audited = {_package(d.get("name", "")) for r in reports for d in r.get("dependencies", [])}
    print("audited %d packages across %d report(s)" % (len(audited), len(reports)))
    for line in warnings:
        print("::warning::" + line)
    for advisory_id in stale:
        print("::notice::%s is accepted but no longer reported; it can leave the list" % advisory_id)
    for line in failures:
        print("::error::" + line)
    print("%d failing, %d accepted and unfixed" % (len(failures), len(warnings)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
