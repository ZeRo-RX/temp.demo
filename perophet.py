import pandas as pd
from prophet import Prophet

# خواندن داده ها از فایل CSV
df = pd.read_csv('data3.csv')

# تبدیل ستون تاریخ به فرمت datetime
df['date'] = pd.to_datetime(df['date'])

# تعداد روزهای آینده که میخواهید پیش‌بینی کنید
num_days_future = 5

# ایجاد داده های ورودی برای پیش بینی
future = pd.DataFrame({'ds': pd.date_range(start=df['date'].max(), periods=num_days_future+1)[1:]})

# تعریف مدل برای TMin
model_tmin = Prophet()
model_tmin.fit(df[['date', 'tmin']].rename(columns={'date': 'ds', 'tmin': 'y'}))

# پیش بینی TMin برای روزهای آینده
tmin_forecast = model_tmin.predict(future)[['ds', 'yhat']].rename(columns={'yhat': 'tmin_predicted'})

# محاسبه درصد خطا برای TMin
tmin_forecast['error_percentage'] = abs((tmin_forecast['tmin_predicted'] - df['tmin'].iloc[-1]) / df['tmin'].iloc[-1] * 100)

# تعریف مدل برای TMax
model_tmax = Prophet()
model_tmax.fit(df[['date', 'tmax']].rename(columns={'date': 'ds', 'tmax': 'y'}))

# پیش بینی TMax برای روزهای آینده
tmax_forecast = model_tmax.predict(future)[['ds', 'yhat']].rename(columns={'yhat': 'tmax_predicted'})

# محاسبه درصد خطا برای TMax
tmax_forecast['error_percentage'] = abs((tmax_forecast['tmax_predicted'] - df['tmax'].iloc[-1]) / df['tmax'].iloc[-1] * 100)

# پرینت نتایج
print("TMin Forecast:")
print(tmin_forecast)
print("TMax Forecast:")
print(tmax_forecast)