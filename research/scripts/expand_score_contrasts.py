"""Generate an ordinal rehearsal bank with deterministic, independently labelled contrasts."""
import argparse
import json
from pathlib import Path
import random


def counted_checks():
    rng = random.Random(20261108)
    domains = [
        ("battery-pack", "battery pack", "insulation", "thermal cutoff"),
        ("lift-platform", "lift platform", "brake hold", "load stability"),
        ("filing-packet", "filing packet", "signature check", "page count"),
        ("gateway-board", "gateway board", "clock stability", "link failover"),
        ("trail-map", "visitor map", "path alignment", "legend review"),
        ("shuttle-trailer", "passenger trailer", "axle alignment", "wheel balance"),
        ("weaving-frame", "weaving frame", "thread tension", "take-up speed"),
        ("relay-module", "relay module", "contact resistance", "coil draw"),
        ("weather-balloon", "weather balloon", "pressure seal", "tracker pairing"),
        ("acoustic-screen", "acoustic screen", "fastener load", "sound attenuation"),
        ("rowing-shell", "rowing shell", "hull inspection", "oarlock inspection"),
        ("valve-block", "valve block", "seat integrity", "actuator travel"),
        ("camera-mount", "camera mount", "stabilizer alignment", "retainer strength"),
        ("paint-batch", "paint batch", "adhesion test", "color comparison"),
        ("turbine-nozzle", "turbine nozzle", "bore diameter", "pressure retention"),
        ("boat-engine", "boat engine", "oil pressure", "coolant circulation"),
    ]
    for slug, item, a, b in domains:
        active_id, retired_id = rng.sample(range(10, 100), 2)
        current, previous = f"{item} {active_id}", f"{item} {retired_id}"
        criteria = [
            f"Neither {a} nor {b} has a passing result for {current}.",
            f"Only one of {a} or {b} has a passing result for {current}.",
            f"Both {a} and {b} have passing results for {current}.",
        ]
        rows = [
            [f"{previous} passed {a} and {b}. The record for {current} contains no results.", 0],
            [f"The work orders for {a} and {b} on {current} are accepted. The report shows neither test has started.", 0],
            [f"Both tests for {current} were completed and marked FAIL. Both passed for {previous}.", 0],
            [f"For {current}, the inspector signed {a} as PASS. The {b} ticket is still waiting for its first result.", 1],
            [f"The {b} certificate marks {current} PASS. The only passing {a} certificate names {previous}.", 1],
            [f"{a.capitalize()} failed on {current}, while {b} passed on that same unit.", 1],
            [f"Two signed result lines identify {current}: {a} PASS and {b} PASS.", 2],
            [f"The old record reports a failed {a} check. The new record lists completed passes for {a} and {b} on {current}.", 2],
            [f"The {b} test was initially pending. It now has a passing result for {current} alongside the confirmed {a} pass.", 2],
            [f"{current} has no {a} result. Two current {b} records for that unit disagree between PASS and CANCELED. Both are equally credible.", .5],
            [f"The {a} result is a confirmed pass for {current}. Two signed records for the same unit disagree about whether {b} passed or failed. Neither is marked final.", 1.5],
            [f"No {b} result exists for {current}. The signed {a} pass lost its unit ID, and both {current} and {previous} were tested.", .5],
            [f"The current unit has a verified {b} pass. Its {a} result is missing an ID and could be the pass for {current} or the earlier unit.", 1.5],
        ]
        yield {"family": "score-contrast-check-" + slug, "type": "score",
               "instructions": f"Score completion for {current} by counting just the two valid passing checks.",
               "criteria": criteria, "rows": rows}


def counted_thresholds():
    rng = random.Random(20261107)
    for unit in ("pages proofed", "successful calls", "boxes inventoried", "kg recovered",
                 "tested ports", "liters dispensed", "accepted seedlings", "minutes online",
                 "cleared invoices", "soldered pins", "validated features", "operational beacons",
                 "labelled samples", "meters installed", "closed tickets", "accepted bids"):
        for instance in range(5):
            lower = rng.randint(5, 85)
            upper = lower + rng.randint(15, 90)
            criteria = [f"Fewer than {lower} {unit}.", f"At least {lower} and below {upper} {unit}.",
                        f"{upper} or more {unit}."]
            values = (rng.randrange(1, lower), rng.randrange(lower, upper), rng.randrange(upper, upper + 50))
            rows = []
            for level, value in enumerate(values):
                distractors = [rng.randrange(value), rng.randrange(value + 1, upper + 51)]
                rng.shuffle(distractors)
                rows += [
                    [f"The previous shift recorded {distractors[0]} {unit}. The verified count for the active shift is {value}.", level],
                    [f"The current confirmed total is {value} {unit}. The work plan projected {distractors[1]}.", level],
                ]
            rows += [
                [f"Two signed counts for the current shift give {lower - 1} and {lower + 2} {unit}. Neither record has precedence.", .5],
                [f"An audit is resolving a disagreement between two current totals, {upper - 2} and {upper + 1} {unit}. Both reports have equal standing.", 1.5],
            ]
            yield {"family": f"score-contrast-threshold-{unit.replace(' ', '-')}-{instance}", "type": "score",
                   "instructions": "Use the verified total for the current shift to select the ordered count range.",
                   "criteria": criteria, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    bank = {"training": [*counted_checks(), *counted_thresholds()], "calibration": []}
    with args.output.open("x", encoding="utf-8", newline="\n") as target:
        json.dump(bank, target, ensure_ascii=False, indent=2)
        target.write("\n")


if __name__ == "__main__":
    main()
