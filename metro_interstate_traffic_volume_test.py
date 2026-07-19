import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.iolib.smpickle as smpickle
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

data_train = pd.read_csv("metro_interstate_traffic_volume/data/data_train_clean.csv")
group_traffic_by_year_week = pd.read_csv("metro_interstate_traffic_volume/data/" + \
    "group_traffic_by_year_week.csv")
data_test = pd.read_csv("metro_interstate_traffic_volume/data/data_test.csv")

data_test["date_time"] = data_test["date_time"].apply(lambda x: np.datetime64(x))

# Create a monthly and a weekday variable for the validation dataset
data_test["Month_dt"] = data_test["date_time"].dt.month
data_test["Weekday_dt"] = data_test["date_time"].dt.weekday
data_test["Hour_dt"] = data_test["date_time"].dt.hour

data_test = data_test.drop(["rain_1h", "snow_1h"], axis=1)

# Check the week data for end of year dates
print(data_test["Year_dt"].value_counts())

print("Week 1, month 12, year 2017 \n", \
    data_test[(data_test["Week_dt"] == 1) & (data_test["Year_dt"] == 2017) & \
        (data_test["Month_dt"] == 12)])

# Temperature
data_test["Temp_celcius"] = data_test["temp"].apply(lambda x: x - 273.15)

# The holiday variable
data_test = data_test.drop(["holiday"], axis=1)

minnesota_calendar = Minnesota()
minnesota_holidays_2017 = minnesota_calendar.holidays(2017)
minnesota_holidays_2018 = minnesota_calendar.holidays(2018)
minnesota_holidays = minnesota_holidays_2017 + minnesota_holidays_2018

minnesota_holidays_dates = [holiday_item[0] for holiday_item in minnesota_holidays]
minnesota_holidays_names = [holiday_item[1] for holiday_item in minnesota_holidays]

data_holidays = pd.DataFrame({"Holiday_date": minnesota_holidays_dates, "Holiday_name": \
    minnesota_holidays_names})
data_holidays = data_holidays[~data_holidays["Holiday_name"].str.endswith("(Observed)")]

print("The first rows in the holiday dataset \n", data_holidays.head(10))
print("The last rows in the holiday dataset \n", data_holidays.tail(10))

# Recreate the holiday variable
data_test["Date_dt"] = data_test["date_time"].dt.date

data_test["Dummy_holiday"] = (data_test["Date_dt"].isin(data_holidays["Holiday_date"].values)) * 1

# Drop the cloud coverage variable
data_test = data_test.drop(["clouds_all"], axis=1)

# Average traffic volume grouped by year, week
print("The first rows in the grouped dataset \n", group_traffic_by_year_week.head())
group_traffic_by_year_week = group_traffic_by_year_week.drop(["traffic_volume_lag"], axis=1)

# Lagged average weeekly traffic volume
test_group_traffic_by_year_week = data_test[["Year_dt", "Week_dt", \
        "traffic_volume"]].groupby(["Year_dt", "Week_dt"])["traffic_volume"].mean()
test_group_traffic_by_year_week = test_group_traffic_by_year_week.reset_index()
test_group_traffic_by_year_week = test_group_traffic_by_year_week.sort_values(by=["Year_dt", "Week_dt"])

print("Last rows in the grouped train dataset \n", group_traffic_by_year_week.tail())
print("First rows in the grouped test dataset \n", test_group_traffic_by_year_week.head())

group_traffic_by_year_week = pd.concat([group_traffic_by_year_week, test_group_traffic_by_year_week], axis=0)
group_traffic_by_year_week["traffic_volume_lag"] = group_traffic_by_year_week["traffic_volume"].shift(1)
group_traffic_by_year_week.dropna()

# Combine the lagged value with the test dataset
data_test = pd.merge(data_test, group_traffic_by_year_week, how="left", on=["Year_dt", "Week_dt"], \
    suffixes=["_final", "_grouped"])
data_test = data_test.drop(["Date_dt", "traffic_volume_grouped"], axis=1)
data_test = data_test.rename({"traffic_volume_final": "traffic_volume"}, axis=1)

# Weather description
data_test["weather_description"] = data_test["weather_description"].replace({"Sky is Clear": "sky is clear"})

# Create dummy variables
data_test = add_dummy_vars(data_test, ["Month_dt", "Weekday_dt", "Hour_dt", "weather_description"])

data_test["Dummy_weather_description_very_heavy_rain"] = 0

cols_base = ["sleet", "shower_drizzle", "shower_snow", "thunderstorm_with_drizzle", "freezing_rain", \
    "thunderstorm_with_light_drizzle", "light_shower_snow", "squalls", "light_rain_and_snow"]
cols_dummies_base = ["Dummy_weather_description_" + var for var in cols_base]
cols_dummies_base = cols_dummies_base + ["Dummy_month_dt_1", "Dummy_weekday_dt_0", "Dummy_hour_dt_0"]

for col in cols_dummies_base:
    try:
        data_test = data_test.drop(col, axis=1)
    except KeyError:
        print("The column {0} is not present in the test dataset".format(col))

cols_train_extra = list(set(data_train.columns) - set(data_test.columns))
cols_test_extra = list(set(data_test.columns) - set(data_train.columns))

print(cols_train_extra)
print(cols_test_extra)

cols_continuous = ["traffic_volume_lag"]
cols_dummies = [col for col in data_test.columns if col.startswith("Dummy")]
cols_dummies_without_weather = [col for col in cols_dummies if (col.startswith("Dummy_month") | \
    col.startswith("Dummy_weekday") | col.startswith("Dummy_hour"))] + ["Dummy_holiday"]

print(cols_dummies_without_weather)

# Fit a linear regression model
y, X = data_train["traffic_volume"], sm.add_constant(data_train[cols_continuous + \
    cols_dummies_without_weather])
res_reg_lin = sm.OLS(y, X).fit()
print(res_reg_lin.summary())

y_test, X_test = data_test["traffic_volume"], sm.add_constant(data_test[cols_continuous + \
    cols_dummies_without_weather])

smpickle.save_pickle(res_reg_lin, "metro_interstate_traffic_volume/models/" + \
    "lin_reg_model_without_weather.pickle")

print("RMSE for the linear regression model on the train dataset \n", \
    root_mse(y, res_reg_lin.predict(X)))
print("RMSE for the linear regression model on the test dataset \n", \
    root_mse(y_test, res_reg_lin.predict(X_test)))
