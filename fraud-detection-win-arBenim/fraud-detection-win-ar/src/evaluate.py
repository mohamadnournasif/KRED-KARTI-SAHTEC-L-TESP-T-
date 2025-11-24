# src/evaluate.py
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)

try:
    from .utils import ensure_dirs, save_json
except ImportError:
    from src.utils import ensure_dirs, save_json


def metrik_sozlugu(y_true, y_pred, y_proba=None):
    md = {
        "kesinlik": float(precision_score(y_true, y_pred)),
        "duyarlilik": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
    }
    if y_proba is not None:
        md["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    return md


def karmasiklik_matrisi(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Normal (0)", "Sahtecilik (1)"],
        yticklabels=["Normal (0)", "Sahtecilik (1)"],
    )
    plt.xlabel("Tahmin")
    plt.ylabel("Gercek")
    plt.title("Karmasiklik Matrisi")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return cm


def roc_ciz(y_true, y_proba, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Egrisi")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return auc


def pr_ciz(y_true, y_proba, out_path):
    p, r, _ = precision_recall_curve(y_true, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(r, p)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Egrisi")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def esik_taramasi(y_true, y_proba, strateji="kesinligi_arttir", min_duyarlilik=0.5):
    p, r, thr = precision_recall_curve(y_true, y_proba)
    thr = np.append(thr, 1.0)

    en_iyi_esik = 0.5
    en_iyi_p = 0.0
    en_iyi_r = 0.0
    en_iyi_f1 = 0.0

    for T, P, R in zip(thr, p, r):
        if strateji == "kesinligi_arttir":
            if R >= min_duyarlilik and P >= en_iyi_p:
                en_iyi_p, en_iyi_r, en_iyi_esik = P, R, T
        else:
            f1 = (2 * P * R) / (P + R) if (P + R) > 0 else 0.0
            if f1 >= en_iyi_f1:
                en_iyi_f1, en_iyi_p, en_iyi_r, en_iyi_esik = f1, P, R, T

    return {
        "esik": float(en_iyi_esik),
        "kesinlik": float(en_iyi_p),
        "duyarlilik": float(en_iyi_r),
    }


def raporu_kaydet(name, y_true, y_pred, y_proba, out_dir):
    ensure_dirs(out_dir)

    report_txt = classification_report(
        y_true, y_pred,
        target_names=["Normal (0)", "Sahtecilik (1)"],
    )
    report_path = os.path.join(out_dir, f"{name}_classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_txt)

    md = metrik_sozlugu(y_true, y_pred, y_proba)
    save_json(md, os.path.join(out_dir, f"{name}_metrics.json"))
    return md, report_path
