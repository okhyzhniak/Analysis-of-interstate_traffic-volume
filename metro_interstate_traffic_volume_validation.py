import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from workalendar.usa import Minnesota

def add_dummy_vars(data, cols_categorical_data):
    for col in cols_categorical_data:
        if data[col].dtype == "object":
            data[col] = data[col].str.replace(" ", "_")
            data[col] = data[col].str.lower()
        data_dummies = pd.get_dummies(data[col], prefix="Dummy_" + str.lower(col)) * 1
        data = pd.concat([data, data_dummies], axis=1)
    return data

def root_mse(y, pred_y):
    diff_sq = np.power(y - pred_y, 2)
    return np.sqrt(np.mean(diff_sq))

data_train = pd.read_csv("metro_interstate_traffic_volume/data/data_train_final.csv")
data_val = pd.read_csv("metro_interstate_traffic_volume/data/data_val.csv")

train_group_traffic_by_year_week = pd.read_csv("metro_interstate_traffic_volume/data/" + \
    "train_group_traffic_by_year_week.csv")

data_train["date_time"] = data_train["date_time"].apply(lambda x: np.datetime64(x))
data_val["date_time"] = data_val["date_time"].apply(lambda x: np.datetime64(x))

# Create a month, weekday and hour variables for the validation dataset
data_val["Month_dt"] = data_val["date_time"].dt.month
data_val["Weekday_dt"] = data_val["date_time"].dt.weekday
data_val["Hour_dt"] = data_val["date_time"].dt.hour

data_val = data_val.drop(["rain_1h", "snow_1h"], axis=1)

# Check the week data for end of year dates
print(data_val["Year_dt"].value_counts())

print("Week 1, month 12, year 2017 \n", \
    data_val[(data_val["Week_dt"] == 1) & (data_val["Year_dt"] == 2017) & \
        (data_val["Month_dt"] == 12)])

# Temperature
data_val["Temp_celcius"] = data_val["temp"].apply(lambda x: x - 273.15)
print("Descriptive statistics for the temperature variable \n", data_val["Temp_celcius"].describe())

# The holiday variable
data_val = data_val.drop(["holiday"], axis=1)

minnesota_calendar = Minnesota()
minnesota_holidays = minnesota_calendar.holidays(2017)

minnesota_holidays_dates = [holiday_item[0] for holiday_item in minnesota_holidays]
minnesota_holidays_names = [holiday_item[1] for holiday_item in minnesota_holidays]

data_holidays = pd.DataFrame({"Holiday_date": minnesota_holidays_dates, "Holiday_name": \
    minnesota_holidays_names})
data_holidays = data_holidays[~data_holidays["Holiday_name"].str.endswith("(Observed)")]

print("The first rows in the holiday dataset \n", data_holidays.head(10))

# Recreate the holiday variable
data_val["Date_dt"] = data_val["date_time"].dt.date

data_val["Dummy_holiday"] = (data_val["Date_dt"].isin(data_holidays["Holiday_date"].values)) * 1

# Drop the cloud coverage variable
data_val = data_val.drop(["clouds_all"], axis=1)

# Lagged average weeekly traffic volume
val_group_traffic_by_year_week = data_val[["Year_dt", "Week_dt", \
        "traffic_volume"]].groupby(["Year_dt", "Week_dt"])["traffic_volume"].mean()
val_group_traffic_by_year_week = val_group_traffic_by_year_week.reset_index()
val_group_traffic_by_year_week = val_group_traffic_by_year_week.sort_values(by=["Year_dt", "Week_dt"])

print("Last rows in the grouped train dataset \n", train_group_traffic_by_year_week.tail())
print("First rows in the grouped validation dataset \n", val_group_traffic_by_year_week.head())

group_traffic_by_year_week = pd.concat([train_group_traffic_by_year_week, \
    val_group_traffic_by_year_week], axis=0)
group_traffic_by_year_week["traffic_volume_lag"] = group_traffic_by_year_week["traffic_volume"].shift(1)
group_traffic_by_year_week.dropna()

# Save grouped traffic values to a separate file
# group_traffic_by_year_week.to_csv("metro_interstate_traffic_volume/data/group_traffic_by_year_week.csv")

# Combine the lagged value with the validation dataset
data_val = pd.merge(data_val, group_traffic_by_year_week, how="left", on=["Year_dt", "Week_dt"], \
    suffixes=["_final", "_grouped"])
data_val = data_val.drop(["Date_dt", "traffic_volume_grouped"], axis=1)
data_val = data_val.rename({"traffic_volume_final": "traffic_volume"}, axis=1)

print("The first rows in the validation dataset \n", data_val.head())
print("The last rows in the validation dataset \n", data_val.tail())

# Weather description
data_val["weather_description"] = data_val["weather_description"].replace({"Sky is Clear": "sky is clear"})

# Create dummy variables
data_train = add_dummy_vars(data_train, ["Month_dt", "Weekday_dt", "Hour_dt", "weather_description"])
data_val = add_dummy_vars(data_val, ["Month_dt", "Weekday_dt", "Hour_dt", "weather_description"])

data_val["Dummy_month_dt_9"] = 0
data_val["Dummy_month_dt_10"] = 0
data_val["Dummy_month_dt_11"] = 0
data_val["Dummy_month_dt_12"] = 0

data_val["Dummy_weather_description_very_heavy_rain"] = 0
data_val["Dummy_weather_description_smoke"] = 0

cols_base = ["sleet", "shower_drizzle", "shower_snow", "thunderstorm_with_drizzle", "freezing_rain", \
    "thunderstorm_with_light_drizzle", "light_shower_snow", "squalls", "light_rain_and_snow"]
cols_dummies_base = ["Dummy_weather_description_" + var for var in cols_base]
cols_dummies_base = cols_dummies_base + ["Dummy_month_dt_1", "Dummy_weekday_dt_0", "Dummy_hour_dt_0"]

data_train = data_train.drop(cols_dummies_base, axis=1)
for col in cols_dummies_base:
    try:
        data_val = data_val.drop(col, axis=1)
    except KeyError:
        print("The column {0} is not present in the validation dataset".format(col))

cols_train_extra = list(set(data_train.columns) - set(data_val.columns))
cols_val_extra = list(set(data_val.columns) - set(data_train.columns))

print(cols_train_extra)
print(cols_val_extra)

cols_continuous = ["traffic_volume_lag"]
cols_dummies = [col for col in data_train.columns if col.startswith("Dummy")]
cols_dummies_without_weather = [col for col in cols_dummies if (col.startswith("Dummy_month") | \
    col.startswith("Dummy_weekday") | col.startswith("Dummy_hour"))] + ["Dummy_holiday"]

# Fit a linear regression model
y, X = data_train["traffic_volume"], sm.add_constant(data_train[cols_continuous + \
    cols_dummies_without_weather])
res_reg_lin = sm.OLS(y, X).fit()
print(res_reg_lin.summary())

# Fit a GLSAR(1) model
res_reg_glsar = sm.GLSAR(y, X, 1).fit()
print(res_reg_glsar.summary())

y_val, X_val = data_val["traffic_volume"], sm.add_constant(data_val[cols_continuous + \
    cols_dummies_without_weather])

print("RMSE for the linear regression model on the train dataset \n", \
    root_mse(y, res_reg_lin.predict(X)))
print("RMSE for the GLSAR(1) regression model on the train dataset \n", \
    root_mse(y, res_reg_glsar.predict(X)))

print("RMSE for the linear regression model on the validation dataset \n", \
    root_mse(y_val, res_reg_lin.predict(X_val)))
print("RMSE for the GLSAR(1) regression model on the validation dataset \n", \
    root_mse(y_val, res_reg_glsar.predict(X_val)))

print("Standard deviation on the validation dataset \n", np.std(y_val))

# Save a final clean train dataset
y = pd.concat([y, y_val], axis=0)
X = pd.concat([X, X_val], axis=0)

data_train_clean = pd.concat([y, X], axis=1)

# data_train_clean.to_csv("metro_interstate_traffic_volume/data/data_train_clean.csv", index=False)
