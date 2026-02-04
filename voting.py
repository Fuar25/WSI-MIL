"""
Result-level fusion (voting) for multi-stain MIL models.
Aggregates predictions/probabilities from different stain models at the result level.
"""

import os
import numpy as np
import pandas as pd
import time
from sklearn.metrics import roc_auc_score

from config.config import runtime_config
from core.fusion_utils import (
    GREEN, YELLOW, RED, CYAN, RESET,
    run_cv_for_stain
)

# =====================
# 写死的多染色配置示例，可按需修改
# =====================
STAIN_RUNS = [
    {
        "name": "HE",
        "data_paths": {
            "positive": "/mnt/5T/GML/Tiff/Experiments/Experiment1/HE/MALT/10x_256px_0px_overlap/features_uni_v2",
            "negative": "/mnt/5T/GML/Tiff/Experiments/Experiment1/HE/Reactive/10x_256px_0px_overlap/features_uni_v2",
        },
        "save_dir": "/mnt/5T/GML/Tiff/Experiments/Experiment1/results/HE",
        "weight": 1.0,
    },
    # {
    #     "name": "CD20",
    #     "data_paths": {
    #         "positive": "/mnt/5T/GML/Tiff/Experiments/Experiment1/CD20/MALT/10x_256px_0px_overlap/features_uni_v2",
    #         "negative": "/mnt/5T/GML/Tiff/Experiments/Experiment1/CD20/Reactive/10x_256px_0px_overlap/features_uni_v2",
    #     },
    #     "save_dir": "/mnt/5T/GML/Tiff/Experiments/Experiment1/results/CD20",
    #     "weight": 1.0,
    # },
    # {
    #     "name": "CD3",
    #     "data_paths": {
    #         "positive": "/mnt/5T/GML/Tiff/Experiments/Experiment1/CD3/MALT/10x_256px_0px_overlap/features_uni_v2",
    #         "negative": "/mnt/5T/GML/Tiff/Experiments/Experiment1/CD3/Reactive/10x_256px_0px_overlap/features_uni_v2",
    #     },
    #     "save_dir": "/mnt/5T/GML/Tiff/Experiments/Experiment1/results/CD3",
    #     "weight": 1.0,
    # },
    {
        "name": "Ki-67",
        "data_paths": {
            "positive": "/mnt/5T/GML/Tiff/Experiments/Experiment1/Ki-67/MALT/10x_256px_0px_overlap/features_uni_v2",
            "negative": "/mnt/5T/GML/Tiff/Experiments/Experiment1/Ki-67/Reactive/10x_256px_0px_overlap/features_uni_v2",
        },
        "save_dir": "/mnt/5T/GML/Tiff/Experiments/Experiment1/results/Ki-67",
        "weight": 1.0,
    },
    # {
    #     "name": "CD21",
    #     "data_paths": {
    #         "positive": "/mnt/5T/GML/Tiff/Experiments/Experiment1/CD21/MALT/10x_256px_0px_overlap/features_uni_v2",
    #         "negative": "/mnt/5T/GML/Tiff/Experiments/Experiment1/CD21/Reactive/10x_256px_0px_overlap/features_uni_v2",
    #     },
    #     "save_dir": "/mnt/5T/GML/Tiff/Experiments/Experiment1/results/CD21",
    #     "weight": 1.0,
    # },
    # {
    #     "name": "CK-pan",
    #     "data_paths": {
    #         "positive": "/mnt/5T/GML/Tiff/Experiments/Experiment1/CK-pan/MALT/10x_256px_0px_overlap/features_uni_v2",
    #         "negative": "/mnt/5T/GML/Tiff/Experiments/Experiment1/CK-pan/Reactive/10x_256px_0px_overlap/features_uni_v2",
    #     },
    #     "save_dir": "/mnt/5T/GML/Tiff/Experiments/Experiment1/results/CK-pan",
    #     "weight": 1.0,
    # },
]

# 融合输出目录
FUSION_SAVE_DIR = "/mnt/5T/GML/Tiff/Experiments/Experiment1/results/Ki-67"

# 跨染色融合策略: "average"(软) 或 "majority"(硬)
CROSS_STAIN_STRATEGY = "majority"


def fuse_binary(big_df, strategy, weights):
    rows = []
    for pid, g in big_df.groupby("patient_id"):
        total_w = g["weight"].sum()
        fused_prob = (g["prob_positive"] * g["weight"]).sum() / total_w
        vote1 = g.loc[g["prediction"] == 1, "weight"].sum()
        vote0 = total_w - vote1
        if strategy == "majority":
            if vote1 > vote0:
                fused_pred = 1
            elif vote1 < vote0:
                fused_pred = 0
            else:
                fused_pred = int(fused_prob >= 0.5)
        else:  # average
            fused_pred = int(fused_prob >= 0.5)
        label_col = "patient_label" if "patient_label" in g.columns else "label"
        label = int(g[label_col].mode().iat[0])
        rows.append({
            "patient_id": pid,
            "fused_prob_positive": fused_prob,
            "fused_prediction": fused_pred,
            "patient_label": label,
            "used_stains": ",".join(sorted(g["stain"].unique())),
        })
    out_df = pd.DataFrame(rows)
    try:
        auc = roc_auc_score(out_df["patient_label"].astype(int), out_df["fused_prob_positive"])
    except Exception as e:
        print(f"Warning: fusion AUC failed: {e}")
        auc = float("nan")
    acc = (out_df["fused_prediction"].astype(int) == out_df["patient_label"].astype(int)).mean()
    return out_df, acc, auc


def fuse_multiclass(big_df, strategy, weights):
    prob_cols = [c for c in big_df.columns if c.startswith("prob_class_")]
    rows = []
    for pid, g in big_df.groupby("patient_id"):
        total_w = g["weight"].sum()
        fused_probs = {
            c: (g[c] * g["weight"]).sum() / total_w for c in prob_cols
        }
        fused_pred = int(np.argmax([fused_probs[c] for c in prob_cols]))
        if strategy == "majority":
            vote_counts = {}
            for cls in g["prediction"].astype(int):
                vote_counts[cls] = vote_counts.get(cls, 0.0) + g.loc[g["prediction"] == cls, "weight"].sum()
            max_vote = max(vote_counts.values())
            top_classes = [cls for cls, v in vote_counts.items() if v == max_vote]
            if len(top_classes) == 1:
                fused_pred = int(top_classes[0])
        label_col = "patient_label" if "patient_label" in g.columns else "label"
        label = int(g[label_col].mode().iat[0])
        row = {"patient_id": pid, "fused_prediction": fused_pred, "patient_label": label, "used_stains": ",".join(sorted(g["stain"].unique()))}
        row.update({c: fused_probs[c] for c in prob_cols})
        rows.append(row)
    out_df = pd.DataFrame(rows)
    try:
        y_true = out_df["patient_label"].astype(int)
        prob_mat = out_df[prob_cols].values
        auc = roc_auc_score(y_true, prob_mat, multi_class="ovr", average="macro")
    except Exception as e:
        print(f"Warning: fusion AUC failed: {e}")
        auc = float("nan")
    acc = (out_df["fused_prediction"].astype(int) == out_df["patient_label"].astype(int)).mean()
    return out_df, acc, auc


def fuse_per_fold(stain_runs, k_folds, fusion_dir, strategy):
    os.makedirs(fusion_dir, exist_ok=True)
    weights = {run["name"]: run.get("weight", 1.0) for run in stain_runs}
    print(f"\n{CYAN}==== Cross-stain fusion: strategy={strategy}, weights={weights} ===={RESET}")
    start_time = time.time()
    fusion_accs = []
    fusion_aucs = []
    fused_dfs = []

    for fold in range(1, k_folds + 1):
        dfs = []
        for run in stain_runs:
            csv_path = os.path.join(run["save_dir"], f"fold_{fold}_patient_{runtime_config.logging.test_results_csv}")
            if not os.path.exists(csv_path):
                print(f"{YELLOW}Warning: missing patient CSV for stain {run['name']} fold {fold}: {csv_path}{RESET}")
                continue
            df = pd.read_csv(csv_path)
            df["stain"] = run["name"]
            df["weight"] = weights.get(run["name"], 1.0)
            dfs.append(df)
        if not dfs:
            print(f"{YELLOW}Fold {fold}: no data to fuse, skipped.{RESET}")
            continue

        big_df = pd.concat(dfs, ignore_index=True)
        if "prob_positive" in big_df.columns:
            fused_df, acc, auc = fuse_binary(big_df, strategy, weights)
        else:
            prob_cols = [c for c in big_df.columns if c.startswith("prob_class_")]
            if not prob_cols:
                print(f"{YELLOW}Fold {fold}: no probability columns found, skipped.{RESET}")
                continue
            fused_df, acc, auc = fuse_multiclass(big_df, strategy, weights)

        fused_df["fold"] = fold
        fused_dfs.append(fused_df)
        fusion_accs.append(acc)
        fusion_aucs.append(auc)

        fusion_path = os.path.join(fusion_dir, f"fold_{fold}_fusion_patient_results.csv")
        fused_df.to_csv(fusion_path, index=False)
        print(f"{GREEN}Fold {fold}: fused saved -> {fusion_path}; patient_acc={acc:.4f}, patient_auc={auc:.4f}{RESET}")

    # 汇总所有折
    if fused_dfs:
        full_df = pd.concat(fused_dfs, ignore_index=True)
        full_path = os.path.join(fusion_dir, "full_fusion_patient_results.csv")
        full_df.to_csv(full_path, index=False)

        # 汇总指标
        mean_auc = np.nanmean(fusion_aucs)
        std_auc = np.nanstd(fusion_aucs)
        mean_acc = np.nanmean(fusion_accs)
        std_acc = np.nanstd(fusion_accs)
        total_time_min = (time.time() - start_time) / 60.0

        print(f"\n{CYAN}Fusion Summary:{RESET}")
        print(f"  {CYAN}Average Fused Patient AUC: {mean_auc:.4f} +/- {std_auc:.4f}{RESET}")
        print(f"  {CYAN}Average Fused Patient Acc: {mean_acc:.4f} +/- {std_acc:.4f}{RESET}")
        print(f"  {CYAN}Total Fusion Time: {total_time_min:.2f} minutes{RESET}")
        print(f"  {GREEN}Full fusion results saved to: {full_path}{RESET}")
        return mean_auc, std_auc, mean_acc, std_acc
    else:
        print(f"{RED}No fusion outputs generated.{RESET}")
        return 0.0, 0.0, 0.0, 0.0


def main():
    k_folds = runtime_config.training.k_folds

    # 逐染色运行 CV (result-level fusion 不需要保存特征)
    for run_cfg in STAIN_RUNS:
        run_cv_for_stain(run_cfg, k_folds, save_features=False)

    # 跨染色按折融合
    return fuse_per_fold(STAIN_RUNS, k_folds, FUSION_SAVE_DIR, CROSS_STAIN_STRATEGY)


if __name__ == "__main__":
    # 可按需覆盖设备等参数
    runtime_config.training.device = "cuda:2"
    runtime_config.training.voting_strategy = "average"  # 单染色患者聚合策略，可调

    import sys
    log_file = "fusion_repeat_log.txt"
    with open(log_file, "w") as f:
        f.write("Repeat Experiment Log\n")

    for i in range(10):
        print(f"\n>>>>>>>>>>>>>>> Repeat {i+1}/10 <<<<<<<<<<<<<<<")
        try:
            mean_auc, std_auc, mean_acc, std_acc = main()
            log_str = (f"Run {i+1}: Average Fused Patient AUC: {mean_auc:.4f} +/- {std_auc:.4f}\n"
                       f"        Average Fused Patient Acc: {mean_acc:.4f} +/- {std_acc:.4f}\n")
            with open(log_file, "a") as f:
                f.write(log_str)
        except Exception as e:
            print(f"Run {i+1} Error: {e}")

