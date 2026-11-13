"""Cache measured offline evidence separately from training and model selection."""

from dataclasses import asdict
import json

from ..dataset import digest, load_manifest, save_json
from ..evaluation_reference import evaluation_cases, load_case_record
from ..provenance import experiment_provenance
from ..rate_audit import perturb_recording
from .audit import SCENARIOS
from .value import Profile, capture_profile


def capture_senssmarttech(dataset, output, *, pilot, development=None, test=None):
    if pilot == bool(development or test) or (not pilot and not (development and test)):
        raise ValueError("Supply both adjudicated packets or explicitly select the published-reference pilot")
    manifest = load_manifest(dataset)
    refs, cases = {}, []
    for split in ("train", "validation", "test"):
        rows, refs[split] = evaluation_cases(dataset, manifest, split, test if split == "test" else development)
        cases.extend(rows)
    config = {"schema": 1, "dataset": manifest["dataset"], "published_reference_pilot": pilot,
              "references": refs, "manifest_sha256": digest(dataset / "manifest.json"), "scenarios": SCENARIOS}
    provenance = experiment_provenance(config)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(provenance) + "\n")
        for index, row in enumerate(cases, 1):
            record = load_case_record(dataset, row)
            original = capture_profile(record)
            for scenario in SCENARIOS:
                profile = original if scenario == "original" else capture_profile(
                    perturb_recording(record, scenario), original=record, original_profile=original)
                item = row | {"subject_id": f"sens:{row['subject_id']}", "dataset": "senssmarttech",
                              "scenario": scenario, "profile": asdict(profile)}
                stream.write(json.dumps(item, allow_nan=False) + "\n")
            if index % 30 == 0 or index == len(cases):
                print(f"SensSmartTech replay: {index}/{len(cases)} recordings", flush=True)
    temporary.replace(output)


def read_replay(path):
    with path.open(encoding="utf-8") as stream:
        provenance = json.loads(next(stream))
        if provenance["configuration"].get("schema") != 1:
            raise ValueError(f"Unsupported replay schema in {path}")
        rows = []
        for line in stream:
            item = json.loads(line)
            rows.append((item, Profile(**item.pop("profile"))))
    if not rows:
        raise ValueError(f"Empty replay: {path}")
    return provenance, rows
