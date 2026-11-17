"""External PCG cycle audit and matched-coverage plotting."""

import csv
import json
from statistics import median

from cardioagent.common.io import digest, save_json
from cardioagent.legacy.multimodal.paired_metrics import risk_coverage, summarize_rates
from cardioagent.legacy.multimodal.provenance import save_provenance


def audit_pcg(source, output):
    from scipy.io import wavfile
    from cardioagent.legacy.multimodal.rate_policy import measure_for_audit

    rows, hashes = [], {}
    for labels in sorted(source.glob("*.tsv")):
        wav = labels.with_suffix(".wav")
        fs, samples = wavfile.read(wav)
        duration = len(samples) / fs
        if samples.ndim != 1:
            raise ValueError(f"Expected a single PCG waveform: {wav.name}")
        with labels.open(newline="", encoding="utf-8-sig") as stream:
            segments = [(float(start), float(end), int(code)) for start, end, code in csv.reader(stream, delimiter="\t")]
        if any(not 0 <= start < end <= duration + 1 / fs for start, end, _ in segments):
            raise ValueError(f"Invalid segmentation timing: {labels.name}")
        periods = [group[-1][0] - group[0][0] for group in zip(*(segments[i:] for i in range(5)))
                   if tuple(segment[2] for segment in group) == (1, 2, 3, 4, 1)
                   and all(abs(a[1] - b[0]) <= .005 for a, b in zip(group, group[1:]))]
        if len(periods) < 3:
            continue
        observed = measure_for_audit("pcg", samples.astype(float), fs, 0.)
        accepted = observed["quality"] == "usable"
        rows.append({"record_id": wav.stem, "subject_id": wav.stem.split("_")[0], "annotated_cycles": len(periods),
                     "reference_bpm": 60 / median(periods), "candidate_bpm": observed["rate_bpm"],
                     "pcg_rate_bpm": observed["rate_bpm"] if accepted else None, "rank": int(accepted),
                     "quality": observed["quality"], "candidate_periods_s": observed["pcg_candidate_periods_s"]})
        hashes.update({file.name: digest(file) for file in (labels, wav)})
    if not rows:
        raise ValueError("No CirCor recordings with three complete annotated S1-to-S1 cycles")
    output.mkdir(parents=True, exist_ok=False)
    config = {"source": "https://physionet.org/content/circor-heart-sound/1.0.3/", "files": hashes,
              "scope": "PCG cycle rate against complete published S1-to-S1 segment sequences",
              "s1_s2_timestamp_evaluation": False}
    provenance = save_provenance(output, config)
    save_json(output / "circor_pcg_period.json", config | {"provenance": provenance, "records": rows,
              "metrics": summarize_rates(rows, "pcg_rate_bpm"), "operating_point": risk_coverage(rows, "pcg_rate_bpm", "rank", [0.])[0]})


def plot_matched(report_path, *, shared=False):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import PercentFormatter

    report = json.loads(report_path.read_text(encoding="utf-8"))
    corrected = report.get("experiment") == "v241"
    version = "V2.4.1" if corrected else "V2.4"
    methods = {"alternate_lead_first": "Alternate ECG first", "channel_quality": "Channel quality",
               "channel_eig": "Channel EIG", "v23_binary_voi": "V2.3 binary VOI", "conditional_voi": f"{version} conditional VOI",
               "conditional_voi_calibrated": f"{version} + calibrated STOP"}
    conditions = {"lead_i_low_snr": "ECG I low SNR", "ecg_ppg_low_snr": "ECG and PPG low SNR",
                  "joint_low_snr": "All modalities low SNR"}
    figure = Figure(figsize=(12.2, 4.65), layout="constrained")
    FigureCanvasAgg(figure)
    figure.suptitle("Selective error at matched coverage · " + ("shared risk score" if shared else "method-specific scores"))
    axes = figure.subplots(1, len(conditions), sharey=True)
    max_risk = .20
    for ax, (scenario, title) in zip(axes, conditions.items()):
        group = report["results"][f"senssmarttech/heart_rate/{scenario}"]["metrics"]
        for i, method in enumerate(methods):
            if method not in group:
                continue
            metric = "shared_risk_operating_points" if shared else "matched_operating_points"
            point = next(p for p in group[method][metric]["fixed_coverage"] if p["target_coverage"] == .70)
            if not point["attainable"]:
                ax.text(.015, i, "Unattainable", color="#666666", va="center", fontsize=8)
                continue
            risk = point["selective_error_rate"]
            ci = point.get("selective_error_rate_ci95")
            ax.scatter(risk, i, color="#205b88" if i < 4 else "#b04827", s=35, zorder=3)
            if ci:
                ax.hlines(i, *ci, color="#62819a" if i < 4 else "#b97c66", linewidth=1.5)
                max_risk = max(max_risk, ci[1])
            max_risk = max(max_risk, risk)
        ax.set_title(title)
        ax.set_yticks(range(len(methods)), labels=list(methods.values()))
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="x", alpha=.2)
    axes[0].invert_yaxis()
    for ax in axes:
        ax.set_xlim(-.02, max_risk * 1.13)
    figure.supxlabel("Error >2 bpm among top-ranked 70% · intervals resample subjects", fontsize=9)
    name = "v241_shared_risk.png" if shared else ("v241_matched_coverage.png" if corrected else "v24_matched_coverage.png")
    figure.savefig(report_path.parent / name, dpi=180)
