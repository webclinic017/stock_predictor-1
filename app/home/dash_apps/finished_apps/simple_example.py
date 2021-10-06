import pandas as pd
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from django_plotly_dash import DjangoDash
from home.algos.us30.us30_model import ai_bot
from home.algos.read_csvs.read_csv import train_model_csv
import datetime


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = DjangoDash('SimpleExample', external_stylesheets=external_stylesheets)


app.layout = html.Div([
    html.H1('Dow Jones Mini Futures Chart vs Predict Prices'),
    dcc.Graph(id='slider-graph', animate=True, style={"backgroundColor": "#1a2d46", 'color': '#ffffff'}),
    dcc.Slider(
        id='slider-updatemode',
        marks={i: '{}'.format(i) for i in range(20)},
        max=20,
        value=2,
        step=1,
        updatemode='drag',
    ),
])


@app.callback(
               Output('slider-graph', 'figure'),
              [Input('slider-updatemode', 'value')])



def display_value(value):


    if int(datetime.datetime.now().minute) % 5 == 0:
        print(int(datetime.datetime.now().minute), "datetime minute")

        bot = ai_bot("YM=F", "ZNZ21.CBT", "ES=F", "5m", -4, None)
        bot.create_data()
        bot.train_model()
        x = bot.data.index[len(bot.Y_train):]
        y = bot.Y_test
        clf_predict_y = bot.clf_predict_y
    else:

        bot = train_model_csv("home/csvs/data_dow_futures.csv", "5m",-4)
        bot.train_model()
        x = bot.data.index[len(bot.Y_train):]
        y = bot.Y_test
        clf_predict_y = bot.clf_predict_y


    graph1 = go.Scatter(
        x= x,
        y= y,
        name='True Dow Prices'
    )
    layout = go.Layout(
        paper_bgcolor='#ffffff',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(range=[min(x), max(x)]),
        yaxis=dict(range=[min(y), max(y)]),
        font=dict(color='black'),

    )

    graph2 = go.Scatter(
        x= x,
        y= clf_predict_y,
        name='Predicted Dow Prices'
    )


    return {'data': [graph1, graph2], 'layout': layout}