import numpy as np
import pandas as pd
from datetime import datetime
import smtplib
import time
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
import pandas_ta
import yfinance as yf
from datetime import date, timedelta
import math
from sklearn.metrics import mean_squared_error
import math
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn import linear_model


class ai_bot():

    def __init__(self, main_stock, stock2, stock3, intervals, time_ahead, start_date):

        self.main_stock = main_stock
        self.stock2 = stock2
        self.stock3 = stock3
        self.intervals = intervals
        self.time_ahead = time_ahead
        self.sart_date = start_date

    def create_data(self):
        df_dow = yf.download(tickers=self.main_stock,  # "^DJI"  "YM=F"
                             interval=self.intervals,
                             start="2021-08-23",
                             end="2021-08-27")

        df_spy = yf.download(tickers=self.stock2,  # ^GSPC  "ES=F"
                             interval=self.intervals,
                             start="2021-08-23",
                             end="2021-08-27")

        df_tnx = yf.download(tickers=self.stock3,  # ^TNX  "ZNZ21.CBT"
                             interval=self.intervals,
                             start="2021-08-23",
                             end="2021-08-27")

        final_df = pd.merge(df_dow, df_tnx[['Close', 'Open', 'High', 'Low']], left_index=True,
                            right_index=True, suffixes=('_dow', '_tnx'))

        final_df = pd.merge(final_df, df_spy[['Close', 'Open', 'High', 'Low']], left_index=True,
                            right_index=True)

        final_df = final_df.rename({'Close': 'Close_spy', 'Open': 'Open_spy', 'High': 'High_spy', 'Low': 'Low_spy'},
                                   axis=1)

        # final_df = pd.merge(df_dow, df_tnx[['Close']], left_index=True, right_index=True, suffixes=('_dow', '_tnx'))

        start_date = date(2021, 8, 30)
        end_date = date.today()
        delta = timedelta(days=7)

        while start_date <= end_date:
            print("start date", start_date.strftime("%Y-%m-%d"))

            end_date2 = start_date + timedelta(days=6)

            print("end date", end_date2)

            df_dow2 = yf.download(tickers=self.main_stock,  # "ESZ21.CME" "^DJI" ^GSPC
                                  interval=self.intervals,
                                  start=start_date.strftime("%Y-%m-%d"),
                                  end=end_date2.strftime("%Y-%m-%d"))

            df_tnx2 = yf.download(tickers=self.stock2,  # ^TNX
                                  interval=self.intervals,
                                  start=start_date.strftime("%Y-%m-%d"),
                                  end=end_date2.strftime("%Y-%m-%d"))

            df_spy2 = yf.download(tickers=self.stock3,
                                  interval=self.intervals,
                                  start=start_date.strftime("%Y-%m-%d"),
                                  end=end_date2.strftime("%Y-%m-%d"))

            print(len(df_spy2), 'len')

            final_df2 = pd.merge(df_dow2, df_tnx2[['Close', 'Open', 'High', 'Low']], left_index=True, right_index=True,
                                 suffixes=('_dow', '_tnx'))

            final_df2 = pd.merge(final_df2, df_spy2[['Close', 'Open', 'High', 'Low']], left_index=True, \
                                 right_index=True)

            final_df2 = final_df2.rename(
                {'Close': 'Close_spy', 'Open': 'Open_spy', 'High': 'High_spy', 'Low': 'Low_spy'}, axis=1)

            # final_df2 = pd.merge(df_dow2, df_tnx2[['Close']], left_index=True, right_index=True, suffixes=('_dow', '_tnx'))

            print(len(final_df2))
            print('-------------')
            final_df = final_df.append(final_df2)

            start_date += delta

        data1 = final_df.copy()

        data1.tail()

        data1.ta.ema(close='Open_dow', length=10, append=True)
        data1.ta.ema(close='Open_dow', length=30, append=True)

        data1['HL_PCT'] = (data1['High_dow'] - data1['Low_dow']) / data1['Close_dow'] * 100.0
        data1['PCT_change'] = (data1['Close_dow'] - data1['Open_dow']) / data1['Open_dow'] * 100.0

        # data1['HL_PCT'] = (data1['High'] - data1['Low']) / data1['Close_dow'] * 100.0
        # data1['PCT_change'] = (data1['Close_dow'] - data1['Open']) / data1['Open'] * 100.0

        data1['HL_PCT_tnx'] = (data1['High_tnx'] - data1['Low_tnx']) / data1['Close_tnx'] * 100.0
        data1['PCT_change_tnx'] = (data1['Close_tnx'] - data1['Open_tnx']) / data1['Open_tnx'] * 100.0

        data1['HL_PCT_spy'] = (data1['High_spy'] - data1['Low_spy']) / data1['Close_spy'] * 100.0
        data1['PCT_change_spy'] = (data1['Close_spy'] - data1['Open_spy']) / data1['Open_spy'] * 100.0

        data1['prediction'] = data1['Open_dow'].shift(self.time_ahead)
        data1.drop(['Adj Close'], axis=1, inplace=True)
        data1.drop(['Volume'], axis=1, inplace=True)

        data1.drop(['Low_tnx'], axis=1, inplace=True)
        data1.drop(['High_tnx'], axis=1, inplace=True)
        data1.drop(['Open_tnx'], axis=1, inplace=True)

        data1.drop(['Low_spy'], axis=1, inplace=True)
        data1.drop(['High_spy'], axis=1, inplace=True)
        data1.drop(['Open_spy'], axis=1, inplace=True)

        print(len(data1), "Length of data")
        print('-------------')

        data1.index = data1.index.strftime('%D %H:%M')

        self.data_for_prediction = data1.copy()  # used to predict latest price

        self.data = data1.copy()

        self.data.dropna(inplace=True)

    def train_model(self):
        forecast_time = 1
        X = np.array(self.data.drop(['prediction'], 1))
        Y = np.array(self.data['prediction'])
        X = preprocessing.scale(X)
        X_prediction = X[-forecast_time:]

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, shuffle=False)

        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test

        # Linear regression
        self.clfreg = LinearRegression(n_jobs=-1)
        self.clfreg.fit(X_train, Y_train)

        # Quadratic Regression 2
        clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
        clfpoly2.fit(X_train, Y_train)

        # Quadratic Regression 3
        clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
        clfpoly3.fit(X_train, Y_train)

        # KNN Regression
        clfknn = KNeighborsRegressor(n_neighbors=2)
        clfknn.fit(X_train, Y_train)

        # Lasso
        clflas = linear_model.Lasso(alpha=0.1)
        clflas.fit(X_train, Y_train)

        # Ridge
        clfrid = linear_model.Ridge(alpha=1.0)
        clfrid.fit(X_train, Y_train)

        confidencereg = self.clfreg.score(X_test, Y_test)
        confidencepoly2 = clfpoly2.score(X_test, Y_test)
        confidencepoly3 = clfpoly3.score(X_test, Y_test)
        confidenceknn = clfknn.score(X_test, Y_test)
        confidencelas = clflas.score(X_test, Y_test)
        confidencerid = clfrid.score(X_test, Y_test)

        print(confidencereg)
        print(confidencepoly2)
        print(confidencepoly3)
        print(confidenceknn)
        print(confidencelas)
        print(confidencerid)
        print('-------------')
        print('-------------')
        print('-------------')
        print('-------------')

        self.clf_predict_y = self.clfreg.predict(self.X_test)
        print('Prediction Score : ', self.clfreg.score(X_test, Y_test))
        error = mean_squared_error(Y_test, self.clf_predict_y)
        print('Mean Squared Error : ', error)
        print("Mean Error L ", math.sqrt(error))
        print('-------------')
        print('-------------')
        print('-------------')
        print('-------------')


    def plot(self):
        # Create traces
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.data.index[len(self.Y_train):], y=self.Y_test,
                                 mode='lines+markers',
                                 name='Y test'))
        fig.add_trace(go.Scatter(x=self.data.index[len(self.Y_train):], y=self.clf_predict_y,
                                 mode='lines+markers',
                                 name='Y predicted'))

        fig.show()

    def predict(self, df):

        num = ""
        for x in self.intervals:
            if x.isdigit():
                num += x

        df2 = df.copy()
        df2.drop(['prediction'], axis=1, inplace=True)
        last_close = df2.iloc[-1]['Open_dow']
        df_test = preprocessing.scale(np.array(df2))
        prediction = self.clfreg.predict(np.array(df_test[-1:]))
        return prediction
        print("Last close price: {} at {}".format(last_close, df2.index[-1]))
        print('Dow prediction in {} minutes: {}'.format(abs(self.time_ahead) * int(num), prediction))
        print('Points move', prediction - last_close)

