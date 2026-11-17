"""Check the frozen adapters and measure sensitivity without fitting a task head."""

import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from cardioagent.legacy.multimodal.benchmark import freeze_assets, select_records
from cardioagent.common.io import digest, save_json
from cardioagent.legacy.multimodal.dataset import load_manifest
from cardioagent.common.recording import MODALITIES, Recording
from cardioagent.legacy.multimodal.workflow import preferred_channel


EXPECTED = {"ecg": [1, 1024], "ppg": [1, 512], "pcg": [1, 3072]}


def run_audit(args, output: Path, models) -> dict:
    manifest = load_manifest(args.dataset_root)
    frozen = freeze_assets(args.models)
    records = select_records(manifest, args.split, args.limit)
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "frozen_assets.json", frozen)
    rng = np.random.default_rng(20261111)
    rows = []
    repeated = None
    for index, row in enumerate(records, 1):
        record = Recording.read(args.dataset_root / "WFDB", row["record_id"])
        checks = {}
        with TemporaryDirectory(dir=output) as temp:
            scratch = Path(temp)
            for m in MODALITIES:
                if m not in record.waves:
                    checks[m] = {"status": "unavailable", "reason": record.unavailable.get(m)}
                    continue
                wave = record.waves[m]
                channel = preferred_channel(m, wave.channels)
                values = wave.window(channel, 0, record.shared_duration)
                clean = scratch / f"{m}.npy"
                check = models.predict(m, channel, values, wave.fs, clean)
                check.pop("embedding_file", None)
                if check["status"] == "ok" and check["embedding_shape"] != EXPECTED[m]:
                    raise ValueError(f"{m} embedding has unexpected shape: {check['embedding_shape']}")
                if m == "ppg" and check["status"] == "ok":
                    if repeated is None:
                        repeat = scratch / "repeat.npy"
                        models.predict(m, channel, values, wave.fs, repeat)
                        repeated = bool(np.array_equal(np.load(clean), np.load(repeat)))
                    # Scale noise relative to the consumed first 10 s, before resampling/normalization.
                    input_values = values[:round(10 * wave.fs)]
                    noise = rng.standard_normal(len(values))
                    rms = np.sqrt(np.mean(noise[:len(input_values)] ** 2))
                    noise *= np.std(input_values) / (math.sqrt(10) * rms)
                    degraded = scratch / "ppg_degraded.npy"
                    perturbed = models.predict(m, channel, values + noise, wave.fs, degraded)
                    if perturbed["status"] == "ok":
                        a, b = (np.load(file).flatten().astype(float) for file in (clean, degraded))
                        norm = np.linalg.norm(a) * np.linalg.norm(b)
                        check["synthetic_10db_cosine"] = float(np.clip(np.dot(a, b) / norm, -1, 1)) if norm > 0 else None
                checks[m] = {"channel": channel, **check}
        rows.append({"record_id": row["record_id"], "checks": checks})
        if index == 1 or index % 25 == 0 or index == len(records):
            print(f"Model audit: {index}/{len(records)}", flush=True)
    cosines = [row["checks"]["ppg"]["synthetic_10db_cosine"] for row in rows
               if row["checks"]["ppg"].get("synthetic_10db_cosine") is not None]
    payload = {
        "dataset": manifest["dataset"], "dataset_manifest_sha256": digest(args.dataset_root / "manifest.json"),
        "split": args.split, "records_audited": len(rows),
        "successful_outputs": {m: sum(r["checks"][m]["status"] == "ok" for r in rows) for m in MODALITIES},
        "expected_embedding_shapes": EXPECTED,
        "anyppg_repeat_identical": repeated,
        "anyppg_perturbation": {
            "type": "additive seeded white noise, 10 dB input SNR, not representative of every artifact",
            "records_scored": len(cosines),
            "median_cosine": float(np.median(cosines)) if cosines else None,
            "p05_cosine": float(np.percentile(cosines, 5)) if cosines else None,
        },
        "records": rows,
    }
    save_json(output / "model_audit.json", payload)
    return payload
