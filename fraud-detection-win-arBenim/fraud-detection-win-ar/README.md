# مشروع كشف الاحتيال (Credit Card Fraud) — Windows + Python 3.9

## المتطلبات
- Windows
- Python 3.9
- صلاحية تثبيت الحزم (`pip`)

## الإعداد السريع (Windows PowerShell)
1) أنشئ بيئة افتراضية وثبّت الاعتمادات:
```
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> إذا واجهت مشكلة مع `xgboost`، جرّب: `pip install xgboost==2.0.3`

2) ضع ملف البيانات داخل: `data/raw/creditcard.csv`
   (إن كان اسم الملف مختلفًا، عدّل `paths.raw_data` داخل `config/config.yaml`)

3) شغّل المعالجة والتقسيم:
```
python main.py preprocess --config config/config.yaml
```

4) درّب نموذج XGBoost + SMOTE (موصى به):
```
python main.py train --model xgb --config config/config.yaml
```

5) قيّم نموذج محفوظ لاحقًا:
```
python main.py evaluate --model-path models/xgboost_smote.pkl --config config/config.yaml
```

ستجد التقارير داخل `outputs/reports/` والرسوم داخل `outputs/figures/`.
