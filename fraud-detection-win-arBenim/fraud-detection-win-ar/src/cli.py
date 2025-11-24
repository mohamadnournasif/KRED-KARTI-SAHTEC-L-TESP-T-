# src/cli.py
import os
import sys
import argparse
import logging
import pandas as pd

# Force UTF-8 on Windows consoles (safe no-op elsewhere)
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from .utils import load_config, setup_logging, ensure_dirs, set_seed
from . import data as dmod
from . import models as mmod
from . import evaluate as ev


def _force_utf8_logging():
    """Ensure logging streams use UTF-8 if possible."""
    try:
        root = logging.getLogger()
        for h in list(root.handlers):
            try:
                if isinstance(h, logging.StreamHandler):
                    h.setStream(sys.stderr)
                    if hasattr(h, "stream") and hasattr(h.stream, "reconfigure"):
                        h.stream.reconfigure(encoding="utf-8")
            except Exception:
                pass
    except Exception:
        pass


# ========================
# Pipeline steps
# ========================

def run_preprocess(cfg):
    paths = cfg["paths"]

    logging.info("Ham veri yukleniyor...")
    df = dmod.load_raw(paths["raw_data"])

    logging.info("On isleme uygulanıyor...")  # if this logs ı issue, change to 'uygulaniyor'
    dfp = dmod.preprocess(df, scale_cols=tuple(cfg["preprocess"]["scale_columns"]))

    logging.info("Hedef sutun tespit ediliyor...")
    target_col = dmod.detect_target(dfp, cfg["preprocess"].get("target", "auto"))

    logging.info("Egitim/Test bolme ve kaydetme islemi...")
    _ = dmod.split_save(
        dfp,
        target_col=target_col,
        train_csv=paths["train_csv"],
        test_csv=paths["test_csv"],
        test_size=cfg["preprocess"]["test_size"],
        random_state=cfg["preprocess"]["random_state"],
        stratify=cfg["preprocess"]["stratify"],
    )

    logging.info("On isleme tamamlandi. Hedef sutun: %s", target_col)


def infer_target_from_train(train_csv: str):
    df = pd.read_csv(train_csv)
    bin_cols = [c for c in df.columns if set(df[c].dropna().unique()).issubset({0, 1})]
    return bin_cols[-1] if bin_cols else None


def run_train(cfg, model: str):
    paths = cfg["paths"]

    target_col = cfg["preprocess"].get("target", "auto")
    if target_col == "auto":
        target_col = infer_target_from_train(paths["train_csv"])

    logging.info("Egitim/Test verileri yukleniyor...")
    X_train, X_test, y_train, y_test = dmod.load_processed(
        paths["train_csv"], paths["test_csv"], target_col
    )

    if model == "iforest":
        logging.info("Isolation Forest egitiliyor...")
        model_obj, mpath = mmod.train_isolation_forest(X_train, cfg, paths["models_dir"])
        y_pred, y_proba = mmod.evaluate_isolation_forest(model_obj, X_test)
        name = "isolation_forest"
        chosen_thr = None

    elif model == "xgb":
        logging.info("XGBoost (SMOTE) egitiliyor...")
        model_obj, mpath = mmod.train_xgboost_smote(X_train, y_train, cfg, paths["models_dir"])

        logging.info("Olasiliklar hesaplaniyor ve esik taramasi yapiliyor...")
        proba = mmod.predict_proba_xgb(model_obj, X_test)
        strat = cfg["thresholds"].get("strategy", "maximize_precision")
        min_r = cfg["thresholds"].get("min_recall", 0.5)

        strategi = "kesinligi_arttir" if strat == "maximize_precision" else "f1"
        sweep = ev.esik_taramasi(y_test, proba, strateji=strategi, min_duyarlilik=min_r)
        chosen_thr = sweep["esik"]

        y_pred, y_proba = mmod.evaluate_xgboost_with_threshold(
            model_obj, X_test, proba_threshold=chosen_thr
        )
        name = "xgboost_smote"

    else:
        raise ValueError("model sunlardan biri olmali: iforest, xgb")

    figures = cfg["paths"]["figures_dir"]
    reports = cfg["paths"]["reports_dir"]
    ensure_dirs(figures, reports)

    cm_path = os.path.join(figures, f"{name}_karmasiklik.png")
    roc_path = os.path.join(figures, f"{name}_roc.png")
    pr_path = os.path.join(figures, f"{name}_pr.png")

    logging.info("Rapor ve grafikler olusturuluyor...")
    md, rpt_path = ev.raporu_kaydet(name, y_test, y_pred, y_proba, reports)
    ev.karmasiklik_matrisi(y_test, y_pred, cm_path)
    if y_proba is not None:
        ev.roc_ciz(y_test, y_proba, roc_path)
        ev.pr_ciz(y_test, y_proba, pr_path)

    logging.info("Model kaydedildi: %s", mpath)
    if chosen_thr is not None:
        logging.info("Secilen olasilik esigi: %.4f", chosen_thr)
    logging.info("Rapor: %s", rpt_path)
    logging.info("Karmasiklik matrisi: %s", cm_path)
    if y_proba is not None:
        logging.info("ROC: %s | PR: %s", roc_path, pr_path)


def run_evaluate(cfg, model_path: str | None):
    paths = cfg["paths"]
    target_col = infer_target_from_train(paths["train_csv"])

    logging.info("Egitim/Test verileri yukleniyor...")
    X_train, X_test, y_train, y_test = dmod.load_processed(
        paths["train_csv"], paths["test_csv"], target_col
    )

    if not model_path:
        model_path = cfg["train"]["model_path"]
        logging.info("--model-path verilmedi, config kullaniliyor: %s", model_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model dosyasi bulunamadi: {model_path}")

    logging.info("Model yukleniyor: %s", model_path)
    model = mmod.load_model(model_path)

    name = os.path.basename(model_path).split(".")[0]
    if "isolation" in name:
        y_pred, y_proba = mmod.evaluate_isolation_forest(model, X_test)
    else:
        thr = cfg["thresholds"].get("default_proba_threshold", 0.5)
        y_pred, y_proba = mmod.evaluate_xgboost_with_threshold(
            model, X_test, proba_threshold=thr
        )

    figures = cfg["paths"]["figures_dir"]
    reports = cfg["paths"]["reports_dir"]
    ensure_dirs(figures, reports)

    cm_path = os.path.join(figures, f"{name}_karmasiklik.png")
    roc_path = os.path.join(figures, f"{name}_roc.png")
    pr_path = os.path.join(figures, f"{name}_pr.png")

    logging.info("Rapor ve grafikler olusturuluyor...")
    md, rpt_path = ev.raporu_kaydet(name, y_test, y_pred, y_proba, reports)
    ev.karmasiklik_matrisi(y_test, y_pred, cm_path)
    if y_proba is not None:
        ev.roc_ciz(y_test, y_proba, roc_path)
        ev.pr_ciz(y_test, y_proba, pr_path)

    logging.info("Rapor: %s", rpt_path)


# ========================
# CLI
# ========================

def main():
    parser = argparse.ArgumentParser(description="Dolandiricilik Tespiti Boru Hatti (Windows/Turkce)")
    # Add --config to main parser so it can appear before subcommand
    parser.add_argument("--config", default="config/config.yaml", help="Yapilandirma dosyasi yolu")

    # Common options shared with subcommands (allows --config after subcommand)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default="config/config.yaml", help="Yapilandirma dosyasi yolu")

    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("preprocess", parents=[common],
                   help="Veriyi yukle, on isle, egitim/test olarak bol ve kaydet")

    ptrain = sub.add_parser("train", parents=[common],
                            help="Modeli egit ve test kumesinde degerlendir")
    ptrain.add_argument("--model", choices=["iforest", "xgb"], required=True, help="Egitilecek model")

    peval = sub.add_parser("evaluate", parents=[common], help="Kayitli bir modeli degerlendir")
    peval.add_argument("--model-path", required=False, help="Model dosyasinin yolu (.joblib/.pkl)")

    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging("config/logging.yaml")
    _force_utf8_logging()

    ensure_dirs(
        cfg["paths"]["outputs_dir"],
        cfg["paths"]["figures_dir"],
        cfg["paths"]["reports_dir"],
        cfg["paths"]["models_dir"],
    )
    set_seed(cfg["preprocess"]["random_state"])

    if args.cmd == "preprocess":
        run_preprocess(cfg)
    elif args.cmd == "train":
        run_train(cfg, args.model)
    elif args.cmd == "evaluate":
        run_evaluate(cfg, args.model_path)


if __name__ == "__main__":
    main()
