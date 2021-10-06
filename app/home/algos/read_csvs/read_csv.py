import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pickle



class train_model_csv():

    def __init__(self, data, intervals, time_ahead):

        self.data = pd.read_csv(data, index_col="Datetime")
        self.intervals = intervals
        self.time_ahead = time_ahead


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

        with open('home/csvs/model.pkl', 'rb') as f:
            clfreg = pickle.load(f)

        self.clf_predict_y = clfreg.predict(self.X_test)



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

        # print("Last close price: {} at {}".format(last_close, df2.index[-1]))
        # print('Dow prediction in {} minutes: {}'.format(abs(self.time_ahead) * int(num), prediction))
        # print('Points move', prediction - last_close)

