import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from binance.client import Client
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Binance istemcisi
client = Client()

# Kullanıcıdan kripto sembolü al
symbol = input("Binance kripto sembolü girin (ör: BTCUSDT): ").upper()

# 1 yıllık tarih aralığı
end_date = datetime.now()
start_date = end_date - timedelta(days=365)

# Binance'ten günlük verileri çek
klines = client.get_historical_klines(
    symbol=symbol,
    interval=Client.KLINE_INTERVAL_1DAY,
    start_str=start_date.strftime("%d %b %Y"),
    end_str=end_date.strftime("%d %b %Y")
)

if not klines or len(klines) < 100:
    print(f"{symbol} için yeterli veri bulunamadı.")
    exit()

# Veri çerçevesi oluştur
df = pd.DataFrame(klines, columns=[
    'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
    'Close_time', 'Quote_asset_volume', 'Number_of_trades',
    'Taker_buy_base', 'Taker_buy_quote', 'Ignore'
])

df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)

# Gerekli sütunlar ve tip dönüşümü
df = df[['Close']].astype(float)
df.rename(columns={'Close': 'Kapanış Fiyatı'}, inplace=True)

# Eksik verileri lineer doldur
df['Kapanış Fiyatı'] = df['Kapanış Fiyatı'].interpolate(method='linear')

# Günlük getiri
daily_returns = df['Kapanış Fiyatı'].pct_change().dropna()

# Monte Carlo simülasyonu
num_simulations = 1000
num_days = 252
last_price = df['Kapanış Fiyatı'].iloc[-1]
simulation_df = pd.DataFrame()

for i in range(num_simulations):
    price_series = [last_price]
    for _ in range(num_days):
        daily_return = np.random.normal(daily_returns.mean(), daily_returns.std())
        price_series.append(price_series[-1] * (1 + daily_return))
    simulation_df[i] = price_series

monte_carlo_forecast = simulation_df.mean(axis=1)
monte_carlo_7days_forecast = monte_carlo_forecast[:7]

# Durağanlık testi
result = adfuller(df['Kapanış Fiyatı'])
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')

differenced = False
if result[1] > 0.05:
    print("Veri sabit değil. Fark alma uygulanıyor.")
    df['Kapanış Fiyatı'] = df['Kapanış Fiyatı'].diff().dropna()
    differenced = True

# ARIMA
arima_model = ARIMA(df['Kapanış Fiyatı'], order=(5, 1, 0))
arima_fitted = arima_model.fit()
arima_forecast_diff = arima_fitted.forecast(steps=7)

# SARIMA
sarima_model = SARIMAX(df['Kapanış Fiyatı'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_fitted = sarima_model.fit(disp=False)
sarima_forecast_diff = sarima_fitted.forecast(steps=7)

if differenced:
    arima_forecast = last_price + arima_forecast_diff.cumsum()
    sarima_forecast = last_price + sarima_forecast_diff.cumsum()
else:
    arima_forecast = arima_forecast_diff
    sarima_forecast = sarima_forecast_diff

# Exponential Smoothing
exp_model = ExponentialSmoothing(df["Kapanış Fiyatı"], trend="add", seasonal=None)
exp_fit = exp_model.fit()
exp_forecast = exp_fit.forecast(steps=7)

# Ridge Regression (son 4 kapanışla)
X, y = [], []
values = df['Kapanış Fiyatı'].values
for i in range(4, len(values)):
    X.append(values[i-4:i])
    y.append(values[i])

X = np.array(X)
y = np.array(y)

# NaN temizliği
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

mask = ~np.isnan(y)
X = X[mask]
y = y[mask]

# Veri setini ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ridge modeli eğitimi ve tahmin
ridge_model = Ridge(alpha=1.0).fit(X_train, y_train)
ridge_forecast = ridge_model.predict(X_test[:7])

# Bayesyen tahmin
posterior_mean = daily_returns.mean()
posterior_std = daily_returns.std()
bayesian_forecast = []
last = last_price
for _ in range(7):
    forecast_return = np.random.normal(posterior_mean, posterior_std)
    next_price = last * (1 + forecast_return)
    bayesian_forecast.append(next_price)
    last = next_price

# Varyans hesapla
daily_variances = [np.var(df['Kapanış Fiyatı'].iloc[:i]) if i > 0 else 0 for i in range(len(df))]

# Tahminleri birleştir
forecast_dates = [datetime.now().date() + timedelta(days=i) for i in range(1, 8)]
results = pd.DataFrame({
    'Tarih': forecast_dates,
    'ARIMA Tahminleri': np.round(arima_forecast.values, 4),
    'SARIMA Tahminleri': np.round(sarima_forecast.values, 4),
    'Exponential Smoothing Tahminleri': np.round(exp_forecast.values, 4),
    'Ridge Regression Tahminleri': np.round(ridge_forecast, 4),
    'Monte Carlo Tahminleri': np.round(monte_carlo_7days_forecast.values, 4),
    'Bayesian Tahminleri': np.round(bayesian_forecast, 4),
    'Varyans': np.round(daily_variances[-7:], 4)
})

# CSV çıktısı
filename = f'{symbol}_binance_tahmin.csv'
results.to_csv(filename, index=False, float_format='%.4f')

print("\nTahmin Sonuçları ve Günlük Varyanslar:")
print(results.head(10))

