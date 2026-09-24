"""Render the Task 7 comparison JSON as Markdown. Every number is taken from the report (no hand-copied values)."""
from __future__ import annotations


def _f(value, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _pct(value) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _summary_cell(summary: dict | None, key: str = "mean") -> str:
    return "n/a" if not summary else _f(summary[key])


def key_findings(report: dict) -> list[str]:
    """Plain-language findings, computed only from the report's numbers."""
    models = report["models"]
    lines: list[str] = []
    for key, m in models.items():
        summary = m["grouped_environment"]["summary"]["roc_auc"]
        gate = m["acceptance_gate_result"]
        outcome = "PASSED" if gate["passed"] else "FAILED (" + ", ".join(gate["failed_criteria"]) + ")"
        lines.append(f"**{key}**: grouped-environment ROC-AUC {_f(summary['mean'])} (worst fold {_f(summary['min'])}); gate {outcome}.")
    best = max(models, key=lambda k: models[k]["grouped_environment"]["summary"]["roc_auc"]["mean"])
    lines.append(f"Best-ranked model by grouped ROC-AUC: **{best}**.")
    for key, m in models.items():
        d = m.get("random_split_diagnostic")
        if d:
            lines.append(
                f"{key}: random-split ROC-AUC {_f(d['random_split_roc_auc'])} vs grouped {_f(d['grouped_environment_mean_roc_auc'])} "
                f"(gap {d['gap_random_minus_grouped']:+.3f}; DIAGNOSTIC ONLY)."
            )
    for key, m in models.items():
        comparison = (m.get("ablations") or {}).get("comparisons", {}).get("history_effect_full_vs_without_history")
        if comparison:
            p = comparison["persistent_thermal"]
            verdict = ("improved" if p["auc_difference"] > 0 else "lowered") if p["ci_excludes_zero"] else "did not measurably change"
            lines.append(
                f"History block, {key}: {verdict} persistent_thermal ROC-AUC (full minus without-history {p['auc_difference']:+.4f}, "
                f"environment-bootstrap 95% CI [{p['bootstrap_95_ci'][0]:+.4f}, {p['bootstrap_95_ci'][1]:+.4f}]); "
                f"overall grouped difference {comparison['overall_grouped']['auc_difference']:+.4f}."
            )
    for key, m in models.items():
        v = m.get("versus_rules_non_sparse_rows")
        if v:
            model, rule = v["model_at_suspected_threshold"], v["rules_suspected_or_confirmed"]
            lines.append(
                f"Versus rules ({key}, non-sparse rows, at its SUSPECTED threshold {_f(m['suspected_operating_point']['threshold'], 2)}): "
                f"hard-negative FPR {_f(model['hard_negative_fpr'])} vs rules {_f(rule['hard_negative_fpr'])} "
                f"({_pct(v['hard_negative_fpr_reduction'])} reduction); recall {_f(model['recall'])} vs {_f(rule['recall'])} "
                f"({v['recall_change']:+.3f}); precision {_f(model['precision'])} vs {_f(rule['precision'])}; F1 {_f(model['f1'])} vs {_f(rule['f1'])}."
            )
    for key, m in models.items():
        c = m["confirmed_probability_analysis"]
        state = "attainable" if c["precision_target_attainable"] else "NOT attainable"
        lines.append(
            f"{key}: probability-only CONFIRMED (precision >= {c['precision_target']:.2f} at recall >= {c['minimum_meaningful_recall']:.2f}) is {state}."
        )
    sel = report["selection"]
    lines.append(f"Selected model: **{sel['selected_model'] or 'NONE'}**" + ("" if sel["selected_model"] else " - no artifact saved."))
    return lines


def render_report_markdown_v5(report: dict) -> str:
    models = report["models"]
    keys = list(models)
    out: list[str] = []
    add = out.append

    add("# Fire Detection ML — V5 Model Comparison (Task 7)\n")
    add(f"> **{report['synthetic_data_notice']}**\n")
    selection = report["selection"]
    add(f"## Verdict\n\n**Selected model: {selection['selected_model'] or 'NONE'}.** {selection['reason']}\n")
    add(_table(
        ["model", "gate", "failed criteria"],
        [[k, "PASS" if models[k]["acceptance_gate_result"]["passed"] else "FAIL",
          ", ".join(models[k]["acceptance_gate_result"]["failed_criteria"]) or "-"] for k in keys],
    ))
    meta = report["meta"]
    add(f"\nDataset `{meta['dataset_version']}` sha256 `{meta['training_csv_sha256']}` ({meta['rows']} rows, "
        f"{len(meta['feature_names'])} canonical features), seed {meta['seed']}; Python {meta['python_version']}, "
        f"scikit-learn {meta['sklearn_version']}, numpy {meta['numpy_version']}, joblib {meta['joblib_version']}.\n")

    add("## Key findings\n")
    for line in key_findings(report):
        add(f"- {line}")
    add("")

    add("## Acceptance gate (fixed before evaluation)\n")
    gate = report["acceptance_gate"]
    add(_table(["criterion", "value"], [[k, str(v)] for k, v in gate.items()]))
    add("\nPrimary evaluation: " + report["primary_evaluation"] + ".\n")

    add("## Gate results per model\n")
    for k in keys:
        criteria = models[k]["acceptance_gate_result"]["criteria"]
        add(f"### {k}\n")
        add(_table(["criterion", "value", "requirement", "passed"],
                   [[name, _f(c["value"], 4), c["requirement"], "yes" if c["passed"] else ("n/a" if c["passed"] is None else "NO")]
                    for name, c in criteria.items()]))
        add("")

    add("## Preprocessing and hyperparameter grid\n")
    for k in keys:
        m = models[k]
        add(f"**{k}** — {m['preprocessing']['imputation']}; scaling: {m['preprocessing']['scaling']}; "
            f"missing indicators: {m['preprocessing']['missing_indicators']}. Fixed: `{m['fixed_hyperparameters']}`. "
            f"Selected by {m['selection_criterion']}: `{m['selected_params']}`.\n")
        add(_table(["params", "grouped ROC-AUC mean", "worst fold", "Brier"],
                   [[str(r["params"]), _f(r["grouped_roc_auc_mean"], 4), _f(r["grouped_roc_auc_min"], 4), _f(r["grouped_brier_mean"], 4)]
                    for r in m["hyperparameter_grid_results"]]))
        add("")

    add("## Grouped-environment results (PRIMARY)\n")
    for k in keys:
        g = models[k]["grouped_environment"]
        add(f"### {k} (threshold 0.50)\n")
        add(_table(["fold", "ROC-AUC", "PR-AUC", "accuracy", "precision", "recall", "F1", "Brier"],
                   [[f["split"], *(_f(f[x]) for x in ("roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1", "brier"))] for f in g["per_fold"]]
                   + [[stat, *(_summary_cell(g["summary"][x], stat) for x in ("roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1", "brier"))]
                      for stat in ("mean", "std", "min", "max")]))
        add("")

    add("## Random split — DIAGNOSTIC ONLY\n")
    rows = []
    for k in keys:
        d = models[k].get("random_split_diagnostic")
        if d:
            rows.append([k, _f(d["random_split_roc_auc"]), _f(d["grouped_environment_mean_roc_auc"]), _f(d["gap_random_minus_grouped"], 4),
                         str(d["environments_present_in_both_train_and_test"])])
    add(_table(["model", "random ROC-AUC", "grouped-env ROC-AUC", "gap", "environments in both sides"], rows))
    add("\nNever used to choose a model.\n")

    add("## Leave-one-regime-out (stress test)\n")
    add("`persistent_thermal` is the only regime with history: holding it out leaves the training set with no multi-pass "
        "examples. That row is a severe out-of-distribution stress test, not a hard criterion.\n")
    for k in keys:
        lor = models[k].get("leave_one_regime_out")
        if not lor:
            continue
        add(f"### {k}\n")
        rows = []
        for regime, entry in lor.items():
            if not isinstance(entry, dict) or "at_thresholds" not in entry:
                continue
            m50 = entry["at_thresholds"]["0.50"]
            rows.append([regime, _f(entry["roc_auc"]), _f(m50["precision"]), _f(m50["recall"]), _f(m50["f1"]), _f(m50["brier"])])
        add(_table(["held-out regime", "ROC-AUC", "precision@0.5", "recall@0.5", "F1@0.5", "Brier"], rows))
        add("")

    add("## Leave-one-no-fire-subtype-out\n")
    for k in keys:
        loso = models[k].get("leave_one_no_fire_subtype_out")
        if not loso:
            continue
        add(f"### {k}\n")
        keys_thr = list(next(iter(loso.values()))["at_thresholds"])
        header = ["held-out subtype", "rows", "mean P(fire)", *(f"FPR@{t}" for t in keys_thr), "high-conf FPR (P>=0.8)"]
        add(_table(header, [[s, str(e["rows"]), _f(e["mean_p_fire"]), *(_f(e["at_thresholds"][t]["false_positive_rate"]) for t in keys_thr),
                             _f(e["high_confidence_false_positive_rate"])] for s, e in loso.items()]))
        add("")

    add("## Paired cases\n")
    for k in keys:
        p = models[k]["paired_cases"]
        add(f"### {k} — overall pairwise ranking accuracy {_f(p['overall']['pairwise_ranking_accuracy'])}, "
            f"mean margin {_f(p['overall']['mean_probability_margin'], 4)}\n")
        add(_table(["pair construction", "pairs", "ranking accuracy", "mean P(fire member) - P(no-fire member)"],
                   [[n, str(e["pairs"]), _f(e["pairwise_ranking_accuracy"]), _f(e["mean_probability_margin"], 4)] for n, e in p["by_pair_type"].items()]))
        add("")

    add("## Sparse early evidence (uncertainty, not accuracy)\n")
    add(_table(
        ["model", "Brier", "ECE", "mean P", "0.35-0.65", "P<0.10", "P>0.90", "fire share when P>0.90"],
        [[k, *(_f(models[k]["sparse_early_evidence"][x]) for x in ("brier_score", "expected_calibration_error", "mean_predicted_p")),
          _pct(models[k]["sparse_early_evidence"]["fraction_uncertain_0_35_to_0_65"]),
          _pct(models[k]["sparse_early_evidence"]["fraction_p_below_0_10"]),
          _pct(models[k]["sparse_early_evidence"]["fraction_p_above_0_90"]),
          _pct(models[k]["sparse_early_evidence"]["observed_fire_fraction_when_p_above_0_90"])] for k in keys]))
    add("\nA constant P = 0.5 has Brier 0.25 on this balanced regime.\n")

    add("## Rule baseline (same rows)\n")
    rule = report["rule_baseline"]
    if rule:
        for variant, label in (("suspected_or_confirmed_is_fire", "SUSPECTED or CONFIRMED = fire"), ("confirmed_only_is_fire", "CONFIRMED only = fire")):
            v = rule[variant]
            add(f"### {label}\n")
            add(_table(["slice", "rows", "precision", "recall", "F1", "FP", "FN", "FPR", "hard-neg FPR"],
                       [[name, str(e["rows"]), _f(e["precision"]), _f(e["recall"]), _f(e["f1"]), str(e["false_positives"]), str(e["false_negatives"]),
                         _f(e["false_positive_rate"]), _f(e["hard_negative_fpr"])]
                        for name, e in [("all rows", v["all_rows"]), ("non-sparse rows", v["non_sparse_rows"]), *v["per_regime"].items()]]))
            add("")
        add("### Model vs rules (non-sparse rows, model at its SUSPECTED threshold)\n")
        add(_table(["model", "threshold", "precision", "recall", "F1", "hard-neg FPR", "FPR reduction vs rules", "recall change"],
                   [[k, _f(models[k]["suspected_operating_point"]["threshold"] if models[k]["suspected_operating_point"] else None, 2),
                     *(_f(models[k]["versus_rules_non_sparse_rows"]["model_at_suspected_threshold"][x]) for x in ("precision", "recall", "f1", "hard_negative_fpr")),
                     _pct(models[k]["versus_rules_non_sparse_rows"]["hard_negative_fpr_reduction"]), _f(models[k]["versus_rules_non_sparse_rows"]["recall_change"])]
                    for k in keys if "versus_rules_non_sparse_rows" in models[k]]))
        add("")

    add("## History ablation (does multi-overpass history help persistent cases?)\n")
    for k in keys:
        ab = models[k].get("ablations")
        if not ab:
            continue
        add(f"### {k}\n")
        add(_table(
            ["feature set", "features", "grouped ROC-AUC", "persistent ROC-AUC", "persistent P@0.5", "persistent R@0.5", "persistent F1@0.5",
             "persistent FPR@0.5", "persistent FPR @ 80% recall"],
            [[name, str(e["feature_count"]), _summary_cell(e["grouped_environment"]["roc_auc"]), _f(e["persistent_thermal"]["roc_auc"]),
              *(_f(e["persistent_thermal"]["at_0_50"][x]) for x in ("precision", "recall", "f1", "false_positive_rate")),
              _f(e["persistent_thermal"]["matched_80_percent_persistent_recall"]["false_positive_rate"])]
             for name, e in ab["results"].items()]))
        add("\nClustered-bootstrap AUC differences (environment resampling):\n")
        add(_table(["comparison", "slice", "AUC difference", "95% CI", "CI excludes 0"],
                   [[label, sl, _f(c["auc_difference"], 4), f"[{c['bootstrap_95_ci'][0]:.4f}, {c['bootstrap_95_ci'][1]:.4f}]", "yes" if c["ci_excludes_zero"] else "no"]
                    for label, parts in ab["comparisons"].items() for sl, c in parts.items()]))
        add("")

    add("## Calibration\n")
    add(_table(["model", "variant", "Brier", "ECE", "grouped ROC-AUC", "adopted"],
               [[k, v, _f(e["brier_score"], 4), _f(e["expected_calibration_error"], 4), _f(e["grouped_roc_auc_mean"], 4),
                 "yes" if models[k]["calibration"]["adopted_variant"] == v else ""]
                for k in keys for v, e in models[k]["calibration"]["variants"].items()]))
    for k in keys:
        add(f"\n- {k}: {models[k]['calibration']['adoption_reason']}")
    add("")

    add("## Threshold analysis\n")
    for k in keys:
        op = models[k]["suspected_operating_point"]
        cf = models[k]["confirmed_probability_analysis"]
        add(f"### {k}\n")
        if op:
            a, ns = op["all_rows"], op["non_sparse_rows"]
            add(f"SUSPECTED candidate threshold **{op['threshold']:.2f}** (highest with >= 80% non-sparse positive recall): "
                f"non-sparse positive recall {_pct(op['non_sparse_positive_recall'])}, non-sparse precision {_f(ns['precision'])}, "
                f"non-sparse FPR {_pct(op['non_sparse_false_positive_rate'])}, hard-negative FPR {_pct(op['hard_negative_fpr'])}, "
                f"all-rows precision {_f(a['precision'])} / recall {_f(a['recall'])}.\n")
            add(_table(["regime", "rows", "precision", "recall", "FPR"],
                       [[r, str(e["rows"]), _f(e["precision"]), _f(e["recall"]), _f(e["false_positive_rate"])] for r, e in op["per_regime"].items()]))
        else:
            add("No threshold reaches 80% non-sparse positive recall.")
        c = cf["candidate"]
        add(f"\nCONFIRMED (probability only): precision >= {cf['precision_target']:.2f} at recall >= {cf['minimum_meaningful_recall']:.2f} is "
            f"**{'attainable' if cf['precision_target_attainable'] else 'NOT attainable'}**"
            + (f" (threshold {c['threshold']:.2f}: precision {_f(c['precision'])}, recall {_f(c['recall'])})." if c else
               f"; best precision at that recall: {_f(cf['highest_precision_at_minimum_recall']['precision']) if cf['highest_precision_at_minimum_recall'] else 'n/a'}"
               + (f" at threshold {cf['highest_precision_at_minimum_recall']['threshold']:.2f}." if cf['highest_precision_at_minimum_recall'] else "."))
            + "\n")

    add("## Corroboration analysis (guardrail candidates for a future policy; nothing implemented)\n")
    for k in keys:
        cor = models[k]["corroboration"]
        add(f"### {k}\n")
        add(_table(["guardrail alone", "rows", "precision", "recall"],
                   [[n, str(e["rows"]), _f(e["precision"]), _f(e["recall"])] for n, e in cor["guardrail_alone"].items()]))
        for threshold, table in cor["with_probability"].items():
            add(f"\nP(fire) >= {threshold}:\n")
            add(_table(["guardrail", "flagged", "precision", "recall", "false positives"],
                       [[n, str(e["flagged_rows"]), _f(e["precision"]), _f(e["recall"]), str(e["false_positives"])] for n, e in table.items()]))
        add("")

    add("## Feature importance and shortcut audit\n")
    for k in keys:
        fi = models[k]["feature_importance"]["grouped_permutation"]
        add(f"### {k} — strongest single feature: `{fi['top_feature']}` with {_pct(fi['top_feature_share'])} of the total positive permutation importance\n")
        add(_table(["feature", "mean AUC drop", "share of positive total"],
                   [[n, _f(e["mean_auc_drop"], 4), _pct(e["share_of_total_positive_drop"])]
                    for n, e in sorted(fi["features"].items(), key=lambda item: item[1]["mean_auc_drop"], reverse=True)[:10]]))
        add("\nBlocks: " + ", ".join(f"{n} {_f(e['mean_auc_drop'], 4)}" for n, e in fi["blocks"].items()) + "\n")
        coef = models[k]["feature_importance"]["logistic_coefficients"]
        if coef:
            add("Standardized coefficients (top 8): " + ", ".join(f"`{c['feature']}` {c['mean_coefficient']:+.3f}" for c in coef["coefficients"][:8]) + "\n")
        tree = models[k]["feature_importance"]["tree_importances"]
        if tree:
            add("Impurity importances (top 8): " + ", ".join(f"`{c['feature']}` {c['mean_importance']:.3f}" for c in tree["importances"][:8]) + "\n")

    add("## Missingness audit\n")
    k0 = keys[0]
    assoc = models[k0]["missingness_audit"]["univariate_association"]
    add(_table(["feature", "missing rate", "fire share when missing", "when present", "indicator AUC", "within-regime indicator AUC"],
               [[n, _pct(e["missing_rate"]), _pct(e["fire_share_when_missing"]), _pct(e["fire_share_when_present"]), _f(e["missingness_indicator_auc"]),
                 ", ".join(f"{r}: {_f(v['missingness_indicator_auc'])}" for r, v in e["within_regime"].items()) or "-"] for n, e in assoc.items()]))
    add("\nModel use (held-out AUC drop when the values vs. the missingness pattern are shuffled):\n")
    for k in keys:
        add(f"- **{k}**: " + "; ".join(f"`{n}` values {e['mean_auc_drop_shuffling_values']:+.4f} / missingness {e['mean_auc_drop_shuffling_missingness']:+.4f}"
                                      for n, e in models[k]["missingness_audit"]["model_use"].items()))
    add("")

    add("## Dataset oracle\n\n" + report["dataset_oracle"] + "\n")
    add("## Limitations\n")
    add("- All training and evaluation data is synthetic; nothing here is real-world wildfire detection accuracy.\n"
        "- Sparse early evidence is intentionally ambiguous; history features exist only in `persistent_thermal`.\n"
        "- No site recurrence, multi-platform diversity or news-source-count signal.\n"
        "- The SUSPECTED threshold is chosen and evaluated on the same grouped out-of-fold probabilities (a single scalar; "
        "per-fold behaviour at that threshold is in the JSON).\n"
        "- V5 is not integrated into runtime.\n")
    return "\n".join(out)
