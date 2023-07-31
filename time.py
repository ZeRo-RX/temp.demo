import pandas as pd

# خواندن داده‌ها از فایل CSV
data = pd.read_csv('data3.csv')

# تبدیل ستون تاریخ به فرمت datetime
data['date'] = pd.to_datetime(data['date'])

# تنظیم ستون تاریخ به عنوان ایندکس داده‌ها
data = data.set_index('date')

# محاسبه Tmin برای هر روز
Tmin = data['tmin'].resample('D').min()

# محاسبه Tmax برای هر روز با استفاده از Tmin
Tmax = data['tmax'].groupby(Tmin.index).max()

# نمایش نتایج
print(Tmin)
print(Tmax)