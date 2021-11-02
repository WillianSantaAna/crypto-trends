import pandas as pd
import numpy as np
import joblib
import mysql.connector
from time import time, sleep
from numbers import Number
from ta import momentum as mo
from ta import trend as tr


def change(x, y):
    if len(x) == len(y):
        change_arr = []
        for i in range(0, len(x)):
            if isinstance(x[i], Number) and isinstance(y[i], Number):
                change_arr.append(((y[i] - x[i]) / x[i]) * 100)
            else:
                change_arr.append('NaN')

        return change_arr
    else:
        print(f"Error: x length {len(x)}, y length {len(y)}")


def build_df(hist_data):
    rsi = mo.rsi(hist_data['Close'])
    tsi = mo.tsi(hist_data['Close'])
    tsi_signal = tr.ema_indicator(tsi, window=13)
    macd = tr.macd(hist_data['Close'])
    macd_diff = tr.macd_diff(hist_data['Close'])
    macd_signal = tr.macd_signal(hist_data['Close'])
    ema_w8 = tr.ema_indicator(hist_data['Close'], window=8)
    ema_w20 = tr.ema_indicator(hist_data['Close'], window=20)
    roc_close_ema_w8 = change(ema_w8, hist_data['Close'])
    roc_ema_w8_ema_w20 = change(ema_w20, ema_w8)

    df = pd.DataFrame({
        'close': np.array(list(hist_data['Close'][50::])),
        'rsi': np.array(list(rsi[50::])),
        'macd': np.array(list(macd[50::])),
        'macd_diff': np.array(list(macd_diff[50::])),
        'macd_signal': np.array(list(macd_signal[50::])),
        'tsi': np.array(list(tsi[50::])),
        'tsi_signal': np.array(list(tsi_signal[50::])),
        # 'roc_close_ema_w8': np.array(list(roc_close_ema_w8[50::])),
        # 'roc_ema_w8_ema_w20': np.array(list(roc_ema_w8_ema_w20[50::])),
        'date': np.array(list(hist_data['Date'][50::]))
    })

    return df


year = 31536000
period1 = round(time() - (year * 0.5))
period2 = round(time())
list_assets = pd.read_csv('data/positive_results.csv')
values = []

try:
    for i in list_assets.values:
        sleep(5)
        hist_data = pd.read_csv(
            f'https://query1.finance.yahoo.com/v7/finance/download/{i[0]}?period1={period1}&period2={period2}&interval=1d&events=history&includeAdjustedClose=true')
        df = build_df(hist_data)

        loaded_rf = joblib.load(f"models/random_forest/{i[0]}_rf.joblib")
        pred = loaded_rf.predict(np.array([df.iloc[:, 0:-1].values[-1]]))

        values.append((i[0], pred[0], df.iloc[:, -1].values[-1]))


    connection = mysql.connector.connect(host='localhost',
                                        database='ml_db',
                                        user='root',
                                        password='a1GhHG5h')
    cursor = connection.cursor()

    sql = f"insert into predicting_trends (symbol, trend, close_date) values (%s, %s, %s)"
    cursor.executemany(sql, values)

    connection.commit()
    connection.close()
except:
    print('error')
