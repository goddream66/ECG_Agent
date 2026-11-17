"""Plot selective outcomes from a completed V2.3 audit without refitting anything."""

import json
from math import ceil
from pathlib import Path

from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter


CONDITIONS = {"original": "Original signals", "joint_low_snr": "Joint low-SNR stress test"}
RULES = {
    "ecg_only": "ECG only", "ppg_fallback": "PPG fallback", "alternate_lead_first": "Alternate ECG first",
    "channel_quality": "Channel quality", "fixed_gaussian_eig": "Fixed Gaussian EIG",
    "channel_eig": "Channel EIG", "cost_aware_channel_eig": "Cost-aware channel EIG",
}
COLORS = {label: colormaps["tab10"](i) for i, label in enumerate(RULES.values())}


def series_for(group: dict, calibrated: bool) -> dict:
    if calibrated:
        return {"Learned VOI · threshold returned estimates": group["calibrated_risk_coverage_on_ungated_voi_outputs"]}
    return {label: group["metrics"][method]["heuristic_selective_risk_coverage"] for method, label in RULES.items()}


def plot_audit(report_path: Path) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    for calibrated in (False, True):
        title = "Validation-calibrated risk–coverage" if calibrated else "Heuristic selective risk–coverage"
        figure = Figure(figsize=(13, 5.1), layout="constrained")
        FigureCanvasAgg(figure)
        suffix = " · ECG-derived reference pilot" if report["published_reference_pilot"] else " · Adjudicated reference"
        figure.suptitle(title + suffix)
        groups = {scenario: report["results"][scenario] for scenario in CONDITIONS}
        curves = {scenario: series_for(group, calibrated) for scenario, group in groups.items()}
        limits = [0.2]
        for scenario, group in groups.items():
            series = list(curves[scenario].values())
            if calibrated:
                series.append(group["metrics"]["learned_voi_calibrated"]["heuristic_selective_risk_coverage"][-1:])
            for points in series:
                limits.extend((point.get("selective_error_rate_ci95") or [point["selective_error_rate"]])[-1]
                              for point in points if point["n"])
        upper = min(1, ceil(10 * max(limits)) / 10)
        for ax, (scenario, scenario_title) in zip(figure.subplots(1, 2), CONDITIONS.items()):
            group = groups[scenario]
            for label, series in curves[scenario].items():
                by_count = {point["n"]: point for point in series if point["n"] > 0}
                points = sorted(by_count.values(), key=lambda point: point["coverage"])
                if not points:
                    continue
                x = [point["coverage"] for point in points]
                y = [point["selective_error_rate"] for point in points]
                line, = ax.plot(x, y, label=label, color=COLORS.get(label, "#2277a8"), marker="o", markersize=4, alpha=0.8)
                for index, point in enumerate(points):
                    ci = point.get("selective_error_rate_ci95")
                    if ci:
                        ax.vlines(x[index], *ci, color=line.get_color(), alpha=0.30 if calibrated else 0.11)
                    if point["low_support"]:
                        ax.scatter(x[index], y[index], color=line.get_color(), marker="x", s=24, zorder=4)
                    if calibrated:
                        ax.annotate(f"{point['n']}/{point['subjects']}", (x[index], y[index]), fontsize=8,
                                    xytext=(0, 9 if index % 2 == 0 else -14), textcoords="offset points", ha="center")
            if calibrated:
                operating = group["metrics"]["learned_voi_calibrated"]
                if operating["selective_error_rate"] is not None:
                    ci = operating["heuristic_selective_risk_coverage"][-1].get("selective_error_rate_ci95")
                    if ci:
                        ax.vlines(operating["coverage"], *ci, color="#ad4728", alpha=0.3)
                    ax.scatter(operating["coverage"], operating["selective_error_rate"], marker="D", s=49,
                               facecolors="none", edgecolors="#ad4728", label="Closed-loop calibrated STOP", zorder=5)
                else:
                    ax.text(0.05, 0.91, "Closed-loop STOP: 0 accepted", transform=ax.transAxes)
                if not any(point["n"] for point in next(iter(curves[scenario].values()))):
                    ax.text(0.05, 0.84, "No returned VOI estimates to threshold", transform=ax.transAxes)
            ax.set(title=scenario_title, xlabel="Coverage", ylabel="Error >2 bpm among accepted estimates",
                   xlim=(-0.035, 1.055), ylim=(-0.05 * upper, 1.12 * upper))
            ax.xaxis.set_major_formatter(PercentFormatter(1))
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.grid(alpha=0.17)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=8, loc="upper right")
        figure.supxlabel("Bars show 95% subject-cluster intervals · × marks low support" +
                          (" · Labels show accepted records / subjects" if calibrated else " · Counts for every threshold are in the audit JSON"),
                          fontsize=9)
        name = "calibrated_risk_coverage.png" if calibrated else "heuristic_risk_coverage.png"
        figure.savefig(report_path.parent / name, dpi=170)
    plot_cost(report, report_path.parent)


def plot_cost(report: dict, output: Path) -> None:
    methods = RULES | {"learned_voi": "Learned VOI", "learned_voi_calibrated": "Calibrated STOP", "oracle": "Oracle"}
    conditions = {"original": "Original signals", "lead_i_low_snr": "ECG I low SNR",
                  "ecg_ppg_low_snr": "All ECG and PPG low SNR", "joint_low_snr": "All modalities low SNR"}
    figure = Figure(figsize=(13, 8.2), layout="constrained")
    FigureCanvasAgg(figure)
    figure.suptitle("Unconditional failure versus replayed analysis cost" +
                    (" · ECG-derived reference pilot" if report["published_reference_pilot"] else " · Adjudicated reference"))
    for ax, (scenario, title) in zip(figure.subplots(2, 2).flat, conditions.items()):
        group = report["results"][scenario]
        metrics = group["metrics"]
        for index, (method, label) in enumerate(methods.items()):
            point = metrics[method]
            ax.scatter(point["mean_replayed_analysis_ms"], point["unconditional_failure"], label=label,
                       color=colormaps["tab10"](index), marker=("o", "v", "s", "P", "D", "^", "X", "h", "<", "*")[index],
                       s=44 if method != "oracle" else 76, alpha=0.86)
        frontier = sorted((metrics[m] for m in group["replayed_time_risk_pareto"]),
                          key=lambda point: point["mean_replayed_analysis_ms"])
        ax.plot([p["mean_replayed_analysis_ms"] for p in frontier], [p["unconditional_failure"] for p in frontier],
                linestyle=":", color="0.35", linewidth=1., label="Non-Oracle Pareto frontier", zorder=0)
        ax.set(title=title, xlabel="Mean replayed waveform and QC time (ms)", ylabel="Refusal or error >2 bpm",
               ylim=(-0.04, 1.04))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=0.17)
    handles, labels = figure.axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncols=4, fontsize=8)
    figure.savefig(output / "voi_risk_cost.png", dpi=170)
