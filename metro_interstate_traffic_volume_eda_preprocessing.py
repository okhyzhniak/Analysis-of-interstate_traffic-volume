import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

from workalendar.usa import Minnesota

data = pd.read_csv("metro_interstate_traffic_volume/data/Metro_Interstate_Traffic_Volume.csv")

data["date_time"] = data["date_time"].apply(lambda x: np.datetime64(x))

print("The shape of the complete dataset \n", data.shape)
print("Data types for the complete dataset \n", data.dtypes)

# Train-test split
data["Week_dt"] = data["date_time"].apply(lambda x: pd.Period(x, "h").week)
data["Year_dt"] = data["date_time"].dt.year

data_train = data[((data["Year_dt"] <= 2017) & (data["Week_dt"] < 35)) | (data["Year_dt"] < 2017)]
data_test = data[((data["Year_dt"] >= 2017) & (data["Week_dt"] >= 35)) | (data["Year_dt"] > 2017)]

print("The shape of the train dataset \n", data_train.shape)
print("The train-test split is clean \n", (data_train.shape[0] + data_test.shape[0]) == data.shape[0])

# data_test.to_csv("metro_interstate_traffic_volume/data/data_test.csv", index=False)

# Train-validation split
data_train_final = data_train[data_train["Year_dt"] <= 2016]
data_val = data_train[data_train["Year_dt"] > 2016]

print("The train-validation split has been correct \n", data_train.shape[0] == (data_train_final.shape[0] + \
    data_val.shape[0]))

# data_val.to_csv("metro_interstate_traffic_volume/data/data_val.csv", index=False)

print("The shape of the final training dataset \n", data_train_final.shape)
print("The first columns of the final training dataset \n", data_train_final.head())

# The histogram and dynamics of the target variable
plt.hist(data_train_final["traffic_volume"], bins=100)
plt.title("The distribution of traffic volume")
plt.xlabel("Traffic volume")
plt.ylabel("Frequency")
plt.show()

max_min_weeks_per_year = \
    data_train_final[["Year_dt", "Week_dt"]].groupby(["Year_dt"])["Week_dt"].agg(["max", "min"])
print("The earliest and the latest week number per year \n", max_min_weeks_per_year)

data_train_final["Month_dt"] = data_train_final["date_time"].dt.month
data_train_final["Weekday_dt"] = data_train_final["date_time"].dt.weekday
data_train_final["Hour_dt"] = data_train_final["date_time"].dt.hour

data_train_final = data_train_final.drop(["rain_1h", "snow_1h"], axis=1)

# Correct the week numbers for end of year dates
print("Week 1, month 12, year 2012 \n", \
    data_train_final[(data_train_final["Week_dt"] == 1) & (data_train_final["Year_dt"] == 2012) & \
        (data_train_final["Month_dt"] == 12)])
print("Week 1, month 12, year 2013 \n", \
    data_train_final[(data_train_final["Week_dt"] == 1) & (data_train_final["Year_dt"] == 2013) & \
        (data_train_final["Month_dt"] == 12)])
print("Week 1, month 12, year 2014 \n", \
    data_train_final[(data_train_final["Week_dt"] == 1) & (data_train_final["Year_dt"] == 2014) & \
        (data_train_final["Month_dt"] == 12)])
print("Week 1, month 12, year 2015 \n", \
    data_train_final[(data_train_final["Week_dt"] == 1) & (data_train_final["Year_dt"] == 2015) & \
        (data_train_final["Month_dt"] == 12)])
print("Week 1, month 12, year 2016 \n", \
    data_train_final[(data_train_final["Week_dt"] == 1) & (data_train_final["Year_dt"] == 2016) & \
        (data_train_final["Month_dt"] == 12)])

data_train_final["Week_dt"][(data_train_final["Week_dt"] == 1) & (data_train_final["Year_dt"] == 2012) & \
        (data_train_final["Month_dt"] == 12)] = 53
data_train_final["Week_dt"][(data_train_final["Week_dt"] == 1) & (data_train_final["Year_dt"] == 2013) & \
    (data_train_final["Month_dt"] == 12)] = 53

print("Week 1, month 12, year 2012 \n", \
    data_train_final[(data_train_final["Week_dt"] == 1) & (data_train_final["Year_dt"] == 2012) & \
        (data_train_final["Month_dt"] == 12)])
print("Week 1, month 12, year 2013 \n", \
    data_train_final[(data_train_final["Week_dt"] == 1) & (data_train_final["Year_dt"] == 2013) & \
        (data_train_final["Month_dt"] == 12)])

# Traffic volume data by year and week
group_traffic_by_year_week = \
    data_train_final[["Year_dt", "Week_dt", \
        "traffic_volume"]].groupby(["Year_dt", "Week_dt"])["traffic_volume"].mean()

plt.plot(list(range(len(group_traffic_by_year_week))), list(group_traffic_by_year_week.values))
plt.title("The dynamics of mean traffic volume")
plt.show()

group_traffic_by_year_week = group_traffic_by_year_week.reset_index()
group_traffic_by_year_week = group_traffic_by_year_week.sort_values(by=["Year_dt", "Week_dt"])
print("The first rows for the mean traffic volume grouped by year and week \n", \
    group_traffic_by_year_week.head())

#group_traffic_by_year_week.to_csv("metro_interstate_traffic_volume/data/" + \
#    "train_group_traffic_by_year_week.csv", index=False)

# Temperature
data_train_final["Temp_celcius"] = data_train_final["temp"].apply(lambda x: x - 273.15)
print("Descriptive statistics for the temperature variable \n", data_train_final["Temp_celcius"].describe())

data_train_final["Temp_celcius"] = data_train_final["Temp_celcius"].replace({-273.15: 0})

plt.hist(data_train_final["Temp_celcius"], bins=100)
plt.title("The distribution of temperature")
plt.xlabel("Temperature")
plt.ylabel("Frequency")
plt.show()

print("Pearson correlation for temperature \n", \
    stats.pearsonr(data_train_final["Temp_celcius"], data_train_final["traffic_volume"]))

# The holiday variable
print("Descriptive statistics for the holiday variable \n", data_train_final["holiday"].describe())
print("Total number of occurences for the holiday variable \n", \
    sum(~data_train_final["holiday"].isna()))

data_train_final = data_train_final.drop(["holiday"], axis=1)

minnesota_calendar = Minnesota()
minnesota_holidays_2012 = minnesota_calendar.holidays(2012)
minnesota_holidays_2013 = minnesota_calendar.holidays(2013)
minnesota_holidays_2014 = minnesota_calendar.holidays(2014)
minnesota_holidays_2015 = minnesota_calendar.holidays(2015)
minnesota_holidays_2016 = minnesota_calendar.holidays(2016)

minnesota_holidays = minnesota_holidays_2012 + minnesota_holidays_2013 + minnesota_holidays_2014 + \
    minnesota_holidays_2015 + minnesota_holidays_2016

minnesota_holidays_dates = [holiday_item[0] for holiday_item in minnesota_holidays]
minnesota_holidays_names = [holiday_item[1] for holiday_item in minnesota_holidays]

data_holidays = pd.DataFrame({"Holiday_date": minnesota_holidays_dates, "Holiday_name": \
    minnesota_holidays_names})
data_holidays = data_holidays[~data_holidays["Holiday_name"].str.endswith("(Observed)")]

print("The first rows in the holiday dataset \n", data_holidays.head(10))
print("The ending of the holiday dataset \n", data_holidays.tail(10))

# Recreate the holiday variable
data_train_final["Date_dt"] = data_train_final["date_time"].dt.date

data_train_final["Dummy_holiday"] = \
    (data_train_final["Date_dt"].isin(data_holidays["Holiday_date"].values)) * 1

group_traffic_by_year_holiday = \
    data_train_final[["Year_dt", "Dummy_holiday", \
        "traffic_volume"]].groupby(["Year_dt", "Dummy_holiday"])["traffic_volume"].mean()

print("Traffic volume by year and holiday \n", group_traffic_by_year_holiday)

# Percentage of cloud cover variable
plt.hist(data_train_final["clouds_all"], bins=100)
plt.title("The distribution of cloud cover")
plt.show()

print("Pearson correlation for cloud coverage \n", \
    stats.pearsonr(data_train_final["clouds_all"], data_train_final["traffic_volume"]))

data_train_final = data_train_final.drop(["clouds_all"], axis=1)

# Weather description
data_train_final["weather_description"] = \
    data_train_final["weather_description"].replace({"Sky is Clear": "sky is clear"})

print("Occurence frequency for the weather description variable \n", \
    data_train_final["weather_description"].value_counts())

# Weather main 
print("Occurence frequency for the weather main variable \n", \
    data_train_final["weather_main"].value_counts())

# Mean traffic volume by weekday
print("Mean traffic volume by weekday \n", \
    data_train_final[["Weekday_dt", "traffic_volume"]].groupby(["Weekday_dt"])["traffic_volume"].mean())

# Mean traffic volume by month
print("Mean traffic volume by month \n", \
    data_train_final[["Month_dt", "traffic_volume"]].groupby(["Month_dt"])["traffic_volume"].mean())

# Mean traffic volume by hour
print("Mean traffic volume by hour \n", \
    data_train_final[["Hour_dt", "traffic_volume"]].groupby(["Hour_dt"])["traffic_volume"].mean())

# Lagged average weeekly traffic volume
group_traffic_by_year_week["traffic_volume_lag"] = group_traffic_by_year_week["traffic_volume"].shift(1)
group_traffic_by_year_week = group_traffic_by_year_week.dropna()

data_train_final = pd.merge(data_train_final, group_traffic_by_year_week, how="left", \
    on=["Year_dt", "Week_dt"], suffixes=["_final", "_grouped"])
data_train_final = data_train_final.dropna()
data_train_final = data_train_final.drop(["Date_dt", "traffic_volume_grouped"], axis=1)
data_train_final = data_train_final.rename({"traffic_volume_final": "traffic_volume"}, axis=1)

print("Pearson correlation for lagged weekly average traffic \n", \
    stats.pearsonr(data_train_final["traffic_volume_lag"], data_train_final["traffic_volume"]))

# data_train_final.to_csv("metro_interstate_traffic_volume/data/data_train_final.csv", index=False)

