import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# خواندن داده‌های CSV
data = pd.read_csv('data3.csv')

# تبدیل ستون date به فرمت datetime
data['date'] = pd.to_datetime(data['date'])

# تنظیم ستون date به عنوان index
data = data.set_index('date')

# ایجاد مدل ARIMA برای پیش‌بینی دما
model_temp = ARIMA(data['tmin'], order=(1, 0, 0))
model_temp_fit = model_temp.fit()

# ایجاد مدل ARIMA برای پیش‌بینی رطوبت
model_humid = ARIMA(data['tmax'], order=(1, 0, 0))
model_humid_fit = model_humid.fit()

# پیش‌بینی دما و رطوبت برای 3 روز آینده
temp_forecast = model_temp_fit.forecast(steps=3)
humid_forecast = model_humid_fit.forecast(steps=3)

# چاپ پیش‌بینی دما و رطوبت برای 3 روز آینده
print('tmin:' , temp_forecast)
print('tmax:', humid_forecast)