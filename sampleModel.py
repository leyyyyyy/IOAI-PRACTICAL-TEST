import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error


# data handling
qc_file = r'D:\Users\Zeniel\Documents\Zenielsffuts\codeffuts\IOAI ML\ioai_qc_train.csv'
qc_data = pd.read_csv(qc_file)
qc_data = qc_data.dropna(axis=0)
# print(qc_data.columns)
y = qc_data.precipitation
features = [
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "dew_point_2m",
    "vapour_pressure_deficit",
    "pressure_msl",
    "surface_pressure",
    "cloud_cover",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation"
]
x = qc_data[features]

# print(x.head)


# model handling
qc_model = DecisionTreeRegressor(random_state=1)
qc_model.fit(x, y)  

print(x.head)
print('predictions')
model_predictions = qc_model.predict(x)
mae = mean_absolute_error(y, model_predictions)
print(mae)