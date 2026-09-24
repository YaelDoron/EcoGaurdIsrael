"""Render the Task 8 confirmatory report as Markdown. Every number comes from the report JSON."""
from __future__ import annotations


def _f(value, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _pct(value) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return "\n".join(lines + ["| " + " | ".join(row) + " |" for row in rows])


def _matrix_rows(entry: dict) -> list[list[str]]:
    rows = []
    for truth, name in (("fire", "FIRE"), ("no_fire", "NO_FIRE")):
        cell = entry[truth]
        rows.append([name, str(cell["rows"]), *(f"{cell[s]} ({_pct(cell['rates'][s])})" for s in ("NO_EVENT", "SUSPECTED", "CONFIRMED"))])
    return rows


def render_policy_report_markdown(report: dict) -> str:
    out: list[str] = []
    add = out.append
    m = report["metrics"]
    gate = report["policy_gate"]

    add("# Fire Detection — AI Hybrid Policy: Confirmatory Evaluation (Task 8)\n")
    add(f"> **{report['synthetic_data_notice']}**\n")
    add(f"## Verdict: **{report['verdict']}**\n")
    add(f"Task 7 binary-primary V5 gate: **{report['task_7_binary_primary_gate']}** (unchanged, historical). Task 8 evaluates a "
        "different architecture: a calibrated HGB likelihood estimator + an explicit uncertainty state (SUSPECTED) + one "
        "deterministic corroboration guardrail. It is not presented as if Task 7 had passed.\n")

    add("## Locked model and policy (fixed before evaluation)\n")
    lm, lp = report["locked_model"], report["locked_policy"]
    add(f"- Model: **{lm['family']}** `{lm['params']}`, the exact ordered V5 feature contract ({len(lm['feature_names'])} features); "
        f"preprocessing: {lm['preprocessing']['imputation']}. Trained once on `{lm['trained_once_on']}` ({lm['training_rows']} rows); "
        f"confirmatory rows used in training: {lm['confirmatory_rows_used_in_training']}.")
    add(f"- Policy `{lp['version']}`: suspect threshold {lp['suspect_threshold']}, confirm threshold {lp['confirm_threshold']}, corroboration: {lp['corroboration']}.")
    for state, rule in lp["rules"].items():
        add(f"  - **{state}**: {rule}")
    add("")

    d = report["confirmatory_dataset"]
    ov = d["overlap_with_training"]
    add("## Confirmatory dataset\n")
    add(_table(["item", "value"], [
        ["seed (training seed 42)", str(d["seed"])], ["rows", str(d["rows"])], ["sha256", f"`{d['sha256']}`"],
        ["environments / pairs", f"{d['environments']} / {d['pairs']}"],
        ["shared environment ids / sample ids / pair ids with training", f"{len(ov['shared_environment_ids'])} / {len(ov['shared_sample_ids'])} / {len(ov['shared_pair_ids'])}"],
        ["exact feature-vector overlap with training (diagnostic)", f"{ov['exact_feature_vector_overlap_rows']} rows ({_pct(ov['exact_feature_vector_overlap_fraction'])}); by regime {ov['exact_overlap_by_regime']}"],
    ]))
    add("")

    add("## Policy acceptance gate (fixed before evaluation)\n")
    add(_table(["criterion", "value", "requirement", "passed"],
               [[n, _f(c["value"], 4), c["requirement"], "yes" if c["passed"] else ("n/a" if c["passed"] is None else "NO")]
                for n, c in gate["criteria"].items()]))
    add(f"\nFailed criteria: {', '.join(gate['failed_criteria']) or 'none'}.\n")

    unc = report.get("gate_metric_uncertainty")
    if unc:
        add("### How robust are the margins? (descriptive; the verdict uses the point estimates)\n")
        add(unc["note"] + "\n")
        rows = []
        for name, entry in unc["metrics"].items():
            point = gate["criteria"][name]["value"] if name in gate["criteria"] else None
            rows.append([name, _f(point, 4), f"[{entry['ci_95'][0]:.4f}, {entry['ci_95'][1]:.4f}]", entry["gate_limit"],
                         "yes" if entry["whole_interval_clears_limit"] else "NO"])
        add(_table(["metric", "point estimate", "95% interval", "gate limit", "whole interval clears limit"], rows))
        add("")

    add("## 3-state outcome matrix\n")
    for title, key in (("All rows", "all_rows"), ("Non-sparse rows", "non_sparse_rows")):
        add(f"### {title}\n")
        add(_table(["ground truth", "rows", "NO_EVENT", "SUSPECTED", "CONFIRMED"], _matrix_rows(report["outcome_matrix"][key])))
        add("")
    add("A false SUSPECTED is operationally far milder than a false CONFIRMED.\n")

    add("## Per-regime outcomes\n")
    rows = []
    for regime, entry in report["per_regime"].items():
        for name, truth in (("FIRE", "fire"), ("NO_FIRE", "no_fire")):
            c = entry[truth]
            rows.append([regime, name, str(c["rows"]), *(f"{c[s]} ({_pct(c['rates'][s])})" for s in ("NO_EVENT", "SUSPECTED", "CONFIRMED"))])
    add(_table(["regime", "truth", "rows", "NO_EVENT", "SUSPECTED", "CONFIRMED"], rows))
    add("")

    add("## Per-latent-subtype outcomes\n")
    rows = []
    for name, e in report["per_latent_subtype"].items():
        s = e["statuses"]
        rows.append([name, "FIRE" if e["label"] else "NO_FIRE", str(e["rows"]), *(f"{s[x]} ({_pct(s['rates'][x])})" for x in ("NO_EVENT", "SUSPECTED", "CONFIRMED")), _f(e["mean_p_fire"])])
    add(_table(["latent subtype", "truth", "rows", "NO_EVENT", "SUSPECTED", "CONFIRMED", "mean P(fire)"], rows))
    add("\nFalse-confirmation rate by no-fire subtype: " + ", ".join(
        f"{n} {_pct(e['confirmed_rate'])}" for n, e in report["per_latent_subtype"].items() if e["label"] == 0) + ".\n")

    sp = report["sparse_early_evidence"]
    add("## Sparse early evidence\n")
    add(f"Brier {_f(sp['brier_score'], 4)} (constant 0.5 = 0.25), ECE {_f(sp['expected_calibration_error'], 4)}, mean P {_f(sp['mean_predicted_p'])}; "
        f"between 0.35–0.65: {_pct(sp['fraction_uncertain_0_35_to_0_65'])}; P<0.10: {_pct(sp['fraction_p_below_0_10'])}; P>0.90: {_pct(sp['fraction_p_above_0_90'])}; "
        f"status rates NO_EVENT {_pct(sp['no_event_rate'])} / SUSPECTED {_pct(sp['suspected_rate'])} / CONFIRMED {_pct(sp['confirmed_rate'])}.\n")
    add("Probability histogram (10 bins 0–1): " + str(sp["probability_histogram_10_bins"]) + "\n")
    add(_table(["ground truth", "rows", "NO_EVENT", "SUSPECTED", "CONFIRMED"], _matrix_rows(sp["status_by_truth"])))
    add("")

    add("## Persistent / history behaviour by satellite_pass_count\n")
    add(report["history_progression_note"] + "\n")
    add(_table(["pass count", "rows", "fire rows", "mean P", "median P", "NO_EVENT", "SUSPECTED", "CONFIRMED", "alert precision", "alert recall", "confirmed precision", "confirmed recall"],
               [[k, str(e["rows"]), str(e["fire_rows"]), _f(e["mean_p_fire"]), _f(e["median_p_fire"]), _pct(e["no_event_rate"]), _pct(e["suspected_rate"]),
                 _pct(e["confirmed_rate"]), _f(e["alert_precision"]), _f(e["alert_recall"]), _f(e["confirmed_precision"]), _f(e["confirmed_recall"])]
                for k, e in report["history_progression_by_pass_count"].items()]))
    add("\nInside `persistent_thermal` only (both labels present):\n")
    add(_table(["pass count", "rows", "mean P (fire rows)", "mean P (no-fire rows)", "confirmed rate (fire)", "confirmed rate (no-fire)", "ROC-AUC"],
               [[k, str(e["rows"]), _f(e["fire_rows_mean_p"]), _f(e["no_fire_rows_mean_p"]), _pct(e["confirmed_rate_fire"]), _pct(e["confirmed_rate_no_fire"]), _f(e["roc_auc"])]
                for k, e in report["history_progression_persistent_regime"].items()]))
    add("")

    rc = report["rule_comparison"]
    add("## Rule detector vs AI Hybrid policy (same confirmatory rows)\n")
    add(rc["definitions"] + "\n")
    add("### Non-sparse rows: alert = SUSPECTED or CONFIRMED\n")
    add(_table(["system", "precision", "recall (fire retained)", "F1", "false positives", "false negatives", "no-fire alert rate", "hard-negative FPR"],
               [[name, *(_f(s[k]) for k in ("precision", "recall", "f1")), str(s["false_positives"]), str(s["false_negatives"]),
                 _f(s["false_positive_rate"]), _f(s["hard_negative_fpr"])] for name, s in (("rule detector", rc["non_sparse_alert"]["rule_detector"]), ("AI hybrid policy", rc["non_sparse_alert"]["ai_hybrid_policy"]))]))
    add("\n### CONFIRMED only\n")
    fc = rc["false_confirmed_rows"]
    add(_table(["system", "confirmed precision (all rows)", "false-confirmed rows", "false-confirmation rate", "non-sparse confirmed recall"],
               [["rule detector", _f(rc["confirmed_precision"]["rule_detector"]), str(fc["rule_detector"]), _pct(fc["rule_detector_rate"]), _f(rc["non_sparse_confirmed_only"]["rule_detector"]["recall"])],
                ["AI hybrid policy", _f(rc["confirmed_precision"]["ai_hybrid_policy"]), str(fc["ai_hybrid_policy"]), _pct(fc["ai_hybrid_policy_rate"]), _f(rc["non_sparse_confirmed_only"]["ai_hybrid_policy"]["recall"])]]))
    add(f"\nRelative reduction of the non-sparse no-fire alert rate versus the rules: {_pct(rc['relative_reduction_in_non_sparse_no_fire_alert_rate'])}.\n")
    add("### Per regime alert behaviour (rule vs policy)\n")
    add(_table(["regime", "rule recall", "policy recall", "rule no-fire alert rate", "policy no-fire alert rate"],
               [[r, _f(rc["rule_per_regime_alert"][r]["recall"]), _f(rc["policy_per_regime_alert"][r]["recall"]),
                 _f(rc["rule_per_regime_alert"][r]["false_positive_rate"]), _f(rc["policy_per_regime_alert"][r]["false_positive_rate"])]
                for r in rc["rule_per_regime_alert"]]))
    add("")

    add("## Calibration and ranking on the confirmatory data\n")
    add(f"ROC-AUC {_f(m['roc_auc'])}, PR-AUC {_f(m['pr_auc'])}, Brier {_f(m['brier_score'], 4)}, ECE {_f(m['expected_calibration_error'], 4)}, "
        f"persistent_thermal ROC-AUC {_f(m['persistent_thermal_roc_auc'])}.\n")
    add(_table(["bin", "count", "mean predicted", "observed fire fraction"],
               [[f"{b['lower']:.1f}-{b['upper']:.1f}", str(b["count"]), _f(b["mean_predicted"]), _f(b["observed_fire_fraction"])] for b in report["calibration_bins"]]))
    add("\n## Limitations\n")
    add("- All training and confirmatory evaluation data is synthetic.\n- Static data cannot prove sequential progression of one event.\n"
        "- The policy is offline: not integrated into runtime; SUSPECTED currently triggers the full response pipeline (see the runtime-integration audit).\n")
    return "\n".join(out)
