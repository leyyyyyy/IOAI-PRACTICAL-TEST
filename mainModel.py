import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, f1_score, precision_score, recall_score

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier
from lightgbm import LGBMRegressor

import matplotlib.pyplot as plt


df = pd.read_csv("ioai_qc_train.csv")   # change filename
testdf = pd.read_csv("TEST_FINAL_INT.csv") 



# dt features
df["datetime"] = pd.to_datetime(df["datetime"])

# df["hour"] = df["datetime"].dt.hour
# df["day"] = df["datetime"].dt.day
df["month"] = df["datetime"].dt.month
df['hour_sin'] = np.sin(2 * np.pi * df["datetime"].dt.hour/24)
df['hour_cos'] = np.cos(2 * np.pi * df["datetime"].dt.hour/24)

df['day_sin'] = np.sin(2 * np.pi * df["datetime"].dt.day/365)
df['day_cos'] = np.cos(2 * np.pi * df["datetime"].dt.day/365)

df['wind_dir_sin'] = np.sin(2 * np.pi * df['wind_direction_10m']/360)
df['wind_dir_cos'] = np.cos(2 * np.pi * df['wind_direction_10m']/360)


def month_to_season(month):
    if month in [11, 12, 1, 2]:
        return 'Amihan'
    elif month in [3, 4, 5]:
        return 'PreSummer'
    else:  # 6,7,8,9,10
        return 'Habagat'

df["season"] = df["datetime"].dt.month.apply(month_to_season)
df = pd.get_dummies(df, columns=["season"])


for lag in [1, 3, 6, 24]:
    df[f"cloud_lag{lag}"] = df["cloud_cover"].shift(lag)
    df[f"pressure_lag{lag}"] = df["pressure_msl"].shift(lag)
    df[f"humidity_lag36hr{lag}"] = df["relative_humidity_2m"].shift(lag)
    df[f"rel_humidity{lag}"] = df["relative_humidity_2m"].shift(lag)
    df[f'precip_lag{lag}'] = df['precipitation'].shift(lag)

drop_cols = [
    "city_name",
    "datetime",
    "rain",
    "snowfall",
    "snow_depth",
    "weather_code",
    "wind_direction_10m",
    "cloud_cover"
]

X = df.drop(columns=drop_cols + ["precipitation"])
y = df["precipitation"]

# fill
X = X.fillna(X.median())



print("feature #: ", len(X.columns))




# # of rain
model_reg = LGBMRegressor(
        n_estimators=2000,
    learning_rate=0.03,
    num_leaves=64,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
    )

# wet hours & dry hours
y_binary = (df["precipitation"].values > 0).astype(int)
model_clf = HistGradientBoostingClassifier(max_iter=200, 
                                     learning_rate=0.1, 
                                     max_depth=5,
                                     random_state=27)


wet_idx = df['precipitation'] > 0
X_wet = df.loc[wet_idx, X.columns]
y_wet = df.loc[wet_idx, 'precipitation']
# print(X_wet)
# print(y_wet)



model_clf.fit(X, y_binary)


rain_proba = model_clf.predict_proba(X)[:, 1]

# empty arr
y_rain_class = (y.values > 0).astype(int)
rain_pred = model_clf.predict(X)
print("F1:", f1_score(y_rain_class, rain_pred),
    "Precision:", precision_score(y_rain_class, rain_pred),
    "Recall:", recall_score(y_rain_class, rain_pred))

# ------------------
# CV
# CV
# CV
tscv = TimeSeriesSplit(n_splits=5)
rmse_scores = []
mae_scores = []
for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    y_cv_pred_reg = np.zeros(len(test_index))

    model_reg.fit(X_train, y_train)
    

    for i in range(len(test_index)):
        if rain_proba[i] >= 0.25:
            # print(model_reg.predict(X_test.iloc[i:i+1])[0])

            y_cv_pred_reg[i] = model_reg.predict(X_test.iloc[i:i+1])[0]
            y_cv_pred_reg[i] = np.clip(y_cv_pred_reg[i], 0, 100)
        else:
            y_cv_pred_reg[i] = 0.0
    rmse = root_mean_squared_error(y_test, y_cv_pred_reg)
    mae = mean_absolute_error(y_test, y_cv_pred_reg)
    mae_scores.append(mae)
    rmse_scores.append(rmse)

print("CV MAE:", np.mean(mae_scores))
print("CV RMSE:", np.mean(rmse_scores))

# -----------------------

model_reg.fit(X, y)
y_pred_reg = np.zeros(len(X))
for i in range(len(X)):
        if rain_proba[i] >= 0.25:
            # print(model_reg.predict(X_test.iloc[i:i+1])[0])

            y_pred_reg[i] = model_reg.predict(X.iloc[i:i+1])[0]
            y_pred_reg[i] = np.clip(y_pred_reg[i], 0, 100)
        else:
            y_pred_reg[i] = 0.0



mae = mean_absolute_error(y, y_pred_reg)
print("mae: ", mae)
# r2 = r2_score(y_test, y_pred_gb)

# print("MAE:", mae)
# print("R²:", r2)

# print(X.describe)

submission = pd.DataFrame({
    "id": np.arange(len(y_pred_reg)),   # or "id" depending on competition
    "precipitation": y_pred_reg
})

submission.to_csv("SUBMISSION_FINAL_INT.csv", index=False)