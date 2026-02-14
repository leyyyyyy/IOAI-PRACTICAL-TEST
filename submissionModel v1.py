import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, f1_score, precision_score, recall_score

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier



df = pd.read_csv("ioai_qc_train.csv")   # change filename
dft = pd.read_csv("TEST_FINAL_INT.csv") 



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


# dt features
dft["datetime"] = pd.to_datetime(dft["datetime"])

# df["hour"] = df["datetime"].dt.hour
# df["day"] = df["datetime"].dt.day
dft["month"] = df["datetime"].dt.month
dft['hour_sin'] = np.sin(2 * np.pi * df["datetime"].dt.hour/24)
dft['hour_cos'] = np.cos(2 * np.pi * df["datetime"].dt.hour/24)

dft['day_sin'] = np.sin(2 * np.pi * df["datetime"].dt.day/365)
dft['day_cos'] = np.cos(2 * np.pi * df["datetime"].dt.day/365)

dft['wind_dir_sin'] = np.sin(2 * np.pi * df['wind_direction_10m']/360)
dft['wind_dir_cos'] = np.cos(2 * np.pi * df['wind_direction_10m']/360)

for lag in [1, 3, 6, 24]:
    df[f'precip_lag{lag}'] = df['precipitation'].shift(lag)
    dft[f'precip_lag{lag}'] = dft['precipitation'].shift(lag)


def month_to_season(month):
    if month in [11, 12, 1, 2]:
        return 'Amihan'
    elif month in [3, 4, 5]:
        return 'PreSummer'
    else:  # 6,7,8,9,10
        return 'Habagat'

df["season"] = df["datetime"].dt.month.apply(month_to_season)
df = pd.get_dummies(df, columns=["season"])

dft["season"] = dft["datetime"].dt.month.apply(month_to_season)
dft = pd.get_dummies(dft, columns=["season"])


# lag
dft["humidity_lag1hr"] = dft["relative_humidity_2m"].shift(1)
dft["humidity_lag24hr"] = dft["relative_humidity_2m"].shift(24)
dft["humidity_lag36hr"] = dft["relative_humidity_2m"].shift(36)

dft["pressure_lag1hr"] = dft["pressure_msl"].shift(1)
dft["pressure_lag24hr"] = dft["pressure_msl"].shift(24)
dft["pressure_lag36hr"] = dft["pressure_msl"].shift(36)

dft["cloud_lag1hr"] = dft["cloud_cover"].shift(1)
dft["cloud_lag24hr"] = dft["cloud_cover"].shift(24)
dft["cloud_lag36hr"] = dft["cloud_cover"].shift(36)

dft["rel_humidity1hr"] = dft["relative_humidity_2m"].shift(1)
dft["rel_humidity24hr"] = dft["relative_humidity_2m"].shift(24)
dft["rel_humidity36hr"] = dft["relative_humidity_2m"].shift(36)

df["humidity_lag1hr"] = df["relative_humidity_2m"].shift(1)
df["humidity_lag24hr"] = df["relative_humidity_2m"].shift(24)
df["humidity_lag36hr"] = df["relative_humidity_2m"].shift(36)

df["pressure_lag1hr"] = df["pressure_msl"].shift(1)
df["pressure_lag24hr"] = df["pressure_msl"].shift(24)
df["pressure_lag36hr"] = df["pressure_msl"].shift(36)

df["cloud_lag1hr"] = df["cloud_cover"].shift(1)
df["cloud_lag24hr"] = df["cloud_cover"].shift(24)
df["cloud_lag36hr"] = df["cloud_cover"].shift(36)

df["rel_humidity1hr"] = df["relative_humidity_2m"].shift(1)
df["rel_humidity24hr"] = df["relative_humidity_2m"].shift(24)
df["rel_humidity36hr"] = df["relative_humidity_2m"].shift(36)





drop_cols = [
    "city_name",
    "datetime",
    "snowfall",
    "snow_depth",
    "weather_code",
    "wind_direction_10m",
    "cloud_cover"
]

X = df.drop(columns=drop_cols + ["precipitation"] + ["rain"])
y = df["precipitation"]


Xt = dft.drop(columns=drop_cols + ["id"])
Xt = Xt.fillna(X.median())

# fill
X = X.fillna(X.median())



print("feature #: ", len(X.columns))



tscv = TimeSeriesSplit(n_splits=5)
rmse_scores = []
mae_scores = []

# # of rain
model_reg = HistGradientBoostingRegressor(
        max_depth=10,
        learning_rate=0.01,
        max_iter=700,
        random_state=27
    )

# wet hours & dry hours
y_binary = (df["precipitation"].values > 0).astype(int)
model_clf = HistGradientBoostingClassifier(max_iter=200, 
                                     learning_rate=0.1, 
                                     max_depth=5,
                                     random_state=27)




model_clf.fit(X, y_binary)
rain_proba = model_clf.predict_proba(Xt)[:, 1]


model_reg.fit(X, y)
y_pred_reg = np.zeros(len(Xt))
for i in range(len(Xt)):
        if rain_proba[i] >= 0.88:


            y_pred_reg[i] = model_reg.predict(Xt.iloc[i:i+1])[0]
            y_pred_reg[i] = np.clip(y_pred_reg[i], 0, 100)
        else:
            y_pred_reg[i] = 0.0


submission = pd.DataFrame({
    "id": np.arange(len(y_pred_reg)),   # or "id" depending on competition
    "precipitation": y_pred_reg
})

submission.to_csv("SUBMISSION_FINAL_INT.csv", index=False)