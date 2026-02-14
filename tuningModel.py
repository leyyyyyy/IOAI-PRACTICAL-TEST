import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("ioai_qc_train.csv")   
testdf = pd.read_csv("TEST_FINAL_INT.csv") 
rmse_scores = []


# dt features
df["datetime"] = pd.to_datetime(df["datetime"])

# df["hour"] = df["datetime"].dt.hour
# df["day"] = df["datetime"].dt.day
df["month"] = df["datetime"].dt.month
df['hour_sin'] = np.sin(2 * np.pi * df["datetime"].dt.hour/24)
df['hour_cos'] = np.cos(2 * np.pi * df["datetime"].dt.hour/24)

df['day_sin'] = np.sin(2 * np.pi * df["datetime"].dt.day/365)
df['day_cos'] = np.cos(2 * np.pi * df["datetime"].dt.day/365)

# df['wind_dir_sin'] = np.sin(2 * np.pi * df['wind_direction_10m']/360)
# df['wind_dir_cos'] = np.cos(2 * np.pi * df['wind_direction_10m']/360)


# lag
# df["precipitation_lag1"] = df["precipitation"].shift(1)
df["humidity_lag1hr"] = df["relative_humidity_2m"].shift(1)
df["pressure_lag1hr"] = df["pressure_msl"].shift(1)
df["cloud_lag1hr"] = df["cloud_cover"].shift(1)



drop_cols = [
    "city_name",
    "datetime",
    "rain",
    "snowfall",
    "snow_depth",
    "weather_code",
    "wind_direction_10m",
    ""
]

X = df.drop(columns=drop_cols + ["precipitation"])
y = df["precipitation"]


X = X.fillna(X.median())



tscv = TimeSeriesSplit(n_splits=30)
for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    X_train['lag_1'] = y_train.shift(1)
    X_test['lag_1'] = y_train.iloc[-1]


# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# param_grid = {
#     "max_iter": [50, 100, 300, 500, 700],
#     "max_depth": [3, 5, 7, 8, 10],
#     "learning_rate": [0.01, 0.05, 0.1, 0.3, 0.4]
# }

model_gb = HistGradientBoostingRegressor(
    max_depth=10,
    learning_rate=0.01,
    max_iter=700,
    random_state=27
)
model_gb.fit(X_train, y_train)
# optimized = GridSearchCV(model_gb, param_grid, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
# optimized.fit(X_train, y_train)
y_pred_gb = model_gb.predict(X_test)
y_pred_gb = np.clip(y_pred_gb, 0, 100)
# print("Best params:", optimized.best_params_)
# print("Best CV RMSE:", -optimized.best_score_)


mae = mean_absolute_error(y_test, y_pred_gb)
r2 = r2_score(y_test, y_pred_gb)

print("MAE:", mae)
print("R²:", r2)
