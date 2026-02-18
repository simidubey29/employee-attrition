import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


# =========================
# PATH SETUP (IMPORTANT)
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "HR_Employee_Attrition.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)


# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATA_PATH)


# =========================
# TARGET ENCODING
# =========================
df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})


# =========================
# DROP UNUSED COLUMNS
# =========================
drop_cols = ["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"]
df.drop(columns=drop_cols, inplace=True, errors="ignore")


# =========================
# ENCODE CATEGORICAL FEATURES
# =========================
for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])


# =========================
# SPLIT FEATURES & TARGET
# =========================
X = df.drop("Attrition", axis=1)
y = df["Attrition"]


# =========================
# TRAIN-TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# =========================
# SCALING
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =========================
# HANDLE CLASS IMBALANCE
# =========================
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(
    X_train_scaled, y_train
)


# =========================
# MODEL TRAINING
# =========================
model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train_resampled, y_train_resampled)


# =========================
# EVALUATION
# =========================
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"ROC-AUC Score: {roc_auc:.4f}")


# =========================
# SAVE MODEL & SCALER
# =========================
joblib.dump(model, os.path.join(MODEL_DIR, "xgboost_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
feature_names = X.columns.tolist()
joblib.dump(feature_names, os.path.join(MODEL_DIR, "feature_names.pkl"))


print("✅ Model and scaler saved successfully!")
