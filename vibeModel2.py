import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, f1_score
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMRegressor

# ======================
# LOAD DATA
# ======================

train = pd.read_csv("ioai_qc_train.csv")
test  = pd.read_csv("TEST_FINAL_INT.csv")

train['datetime'] = pd.to_datetime(train['datetime'])
test['datetime']  = pd.to_datetime(test['datetime'])

train = train.sort_values('datetime')
test  = test.sort_values('datetime')

# ======================
# TARGET + CLASS LABEL
# ======================

TARGET = "precipitation"
train['rain_binary'] = (train[TARGET] > 0).astype(int)

# ======================
# LAG FEATURES
# ======================

lags = [1, 3, 6, 12, 24]

for lag in lags:
    train[f'precip_lag_{lag}'] = train[TARGET].shift(lag)

train.dropna(inplace=True)

# ---- FIX FOR TEST DATA ---- #

# Grab last known precipitation values from train
last_values = train[TARGET].iloc[-max(lags):].values

for lag in lags:
    test[f'precip_lag_{lag}'] = last_values[-lag]


# ======================
# SEASON FEATURES (PHILIPPINES)
# ======================

def ph_season(month):
    if month in [12,1,2,3,4,5]:
        return 0   # Dry
    else:
        return 1   # Wet

train['season'] = train['datetime'].dt.month.map(ph_season)
test['season']  = test['datetime'].dt.month.map(ph_season)

# ======================
# FEATURES
# ======================

drop_cols = ['datetime', TARGET, 'rain_binary', 'rain', 'city_name']
features = [c for c in train.columns if c not in drop_cols]

X = train[features]
y_class = train['rain_binary']
y_reg   = train[TARGET]

X_test = test[features]

# ======================
# STEP 1 — RAIN CLASSIFIER
# ======================

clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=20,
    random_state=42,
    n_jobs=-1
)

tscv = TimeSeriesSplit(n_splits=5)

f1_scores = []

for train_idx, val_idx in tscv.split(X):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y_class.iloc[train_idx], y_class.iloc[val_idx]

    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_val)

    f1_scores.append(f1_score(y_val, preds))

print("CV F1:", np.mean(f1_scores))

clf.fit(X, y_class)

test_rain_prob = clf.predict_proba(X_test)[:,1]
test_rain_flag = (test_rain_prob >= 0.5).astype(int)

# ======================
# STEP 2 — REGRESSION MODEL
# ======================

reg = LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=64,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

reg.fit(X, y_reg)

test_reg_preds = reg.predict(X_test)

# ======================
# COMBINE OUTPUTS
# ======================

final_preds = test_reg_preds * test_rain_flag
final_preds = np.clip(final_preds, 0, None)

# ======================
# CREATE ID COLUMN
# ======================

submission = pd.DataFrame({
    "id": np.arange(len(test)),
    "precipitation": final_preds
})

submission.to_csv("submission.csv", index=False)

print("submission.csv created successfully")
