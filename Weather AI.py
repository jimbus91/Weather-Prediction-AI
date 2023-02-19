import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# load the csv file into a pandas dataframe, which can be found on:
# https://www.ncei.noaa.gov/access/us-climate-normals/#dataset=normals-daily&timeframe=15
df = pd.read_csv('normals-daily-2006-2020-2023-02-03T18-55-52.csv')

# convert the 'DATE' column to a datetime object
df['DATE'] = pd.to_datetime(df['DATE'], format='%d-%b')

# extract the columns for temperature
temps = ['DLY-TMIN-NORMAL', 'DLY-TAVG-NORMAL', 'DLY-TMAX-NORMAL']

# prepare the data for modeling
X = df[temps].values
y = df[['DLY-TAVG-NORMAL']].values

# split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# train a linear regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# make predictions on the next year of temperature data
X_pred = df[temps].values[-364:]
pred_temps = regressor.predict(X_pred)
pred_min = X_pred[:, 0]
pred_max = X_pred[:, 2]

# Set the style to dark theme
plt.style.use('dark_background')

today = df['DATE'].iloc[-1]

# Adjust the TimeDelta to match todays date
one_year_ago = today - pd.Timedelta(316, unit='d')
two_years_from_now = today + pd.Timedelta(77, unit='d')

# Add 38 days to the end of the plot
start_date = max(one_year_ago, df['DATE'].iloc[0])
end_date = min(two_years_from_now + pd.Timedelta(38, unit='d'), df['DATE'].iloc[-1])

# plot the actual maximum temperature
plt.plot(df['DATE'][(df['DATE'] >= start_date) & (df['DATE'] <= end_date)], df['DLY-TMAX-NORMAL'][(df['DATE'] >= start_date) & (df['DATE'] <= end_date)], label='Maximum', color='red')

# plot the actual average temperature
plt.plot(df['DATE'][(df['DATE'] >= start_date) & (df['DATE'] <= end_date)], df['DLY-TAVG-NORMAL'][(df['DATE'] >= start_date) & (df['DATE'] <= end_date)], label='Average', color='green')

# plot the actual minimum temperature
plt.plot(df['DATE'][(df['DATE'] >= start_date) & (df['DATE'] <= end_date)], df['DLY-TMIN-NORMAL'][(df['DATE'] >= start_date) & (df['DATE'] <= end_date)], label='Minimum', color='blue')

# get the dates for the predicted temperatures
date_pred = pd.date_range(today + pd.Timedelta(60), periods=364)

# plot the predicted maximum temperature
plt.plot(date_pred, pred_max, label='Predicted', color='magenta')

# plot the predicted average temperature
plt.plot(date_pred, pred_temps.flatten(), label='', color='magenta')

# plot the predicted minimum temperature
plt.plot(date_pred, pred_min, label='', color='magenta')

plt.xlim(one_year_ago, two_years_from_now)

plt.xticks(rotation=45)
plt.ylabel('Temperature (°F)')
plt.title('Colorado Springs Weather Prediction')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.legend()
plt.show()