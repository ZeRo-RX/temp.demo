import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import  mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Reading data from CSV file
df = pd.read_csv('data3.csv')

# Converting date column to datetime format
df['date'] = pd.to_datetime(df['date'])

# Extracting year, month, day, and dayofweek as new features
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek

# Preparing input data
X = df[['year', 'month', 'day', 'dayofweek']]
y_tmin = df['tmin']
y_tmax = df['tmax']

# Splitting data into train andtest sets
X_train, X_test, y_tmin_train, y_tmin_test, y_tmax_train, y_tmax_test = train_test_split(X, y_tmin, y_tmax, test_size=0.2, random_state=42)

# Training model for TMin using XGBoost
model_tmin = xgb.XGBRegressor()
model_tmin.fit(X_train, y_tmin_train)

# Training model for TMax using XGBoost
model_tmax = xgb.XGBRegressor()
model_tmax.fit(X_train, y_tmax_train)

# Predicting TMin and TMax for future days
num_days_future = 1000
future = pd.DataFrame({'date': pd.date_range(start=df['date'].max(), periods=num_days_future+1)[1:]})
future['year'] = future['date'].dt.year
future['month'] = future['date'].dt.month
future['day'] = future['date'].dt.day
future['dayofweek'] = future['date'].dt.dayofweek
future_x = future[['year', 'month', 'day', 'dayofweek']]

tmin_forecast = model_tmin.predict(future_x)
tmax_forecast = model_tmax.predict(future_x)

# Evaluating the model with test data
tmin_test_forecast = model_tmin.predict(X_test)
tmax_test_forecast = model_tmax.predict(X_test)
tmin_error_percentage = mean_absolute_error(y_tmin_test, tmin_test_forecast) / y_tmin_test.mean() * 100
tmax_error_percentage = mean_absolute_error(y_tmax_test, tmax_test_forecast) / y_tmax_test.mean() * 100

# Preparing new DataFrame to display forecasts along with date and error percentage
forecast_df = pd.DataFrame({'Date': future['date'], 'TMin Forecast': tmin_forecast, 'TMax Forecast': tmax_forecast})

# Printing the error percentage
print(f"TMin Error Percentage: {tmin_error_percentage}%")
print(f"TMax Error Percentage: {tmax_error_percentage}%")

# Plotting the actual and predicted values
plt.figure(figsize=(10, 6))
plt.plot(df['date'], df['tmin'], label='Actual TMin')
plt.plot(future['date'], tmin_forecast, label='Predicted TMin')
plt.legend()
plt.xlabel('Date')
plt.ylabel('TMin')
plt.title('Actual vs. Predicted TMin')
plt.show()
