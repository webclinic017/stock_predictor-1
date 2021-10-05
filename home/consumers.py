from channels.generic.websocket import AsyncWebsocketConsumer
import json
from random import randint
import time
from home.algos.us30.us30_model import ai_bot
from home.algos.read_csvs.read_csv import train_model_csv
import datetime

from asyncio import sleep

class GraphConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()


        while True:
            if int(datetime.datetime.now().minute) % 2 == 0:
                print(int(datetime.datetime.now().minute), "datetime minute")

                bot = ai_bot("YM=F", "ZNZ21.CBT", "ES=F", "5m", -4, None)
                bot.create_data()
                bot.train_model()
                predict = bot.predict(bot.data_for_prediction)
                print(predict, 'the_prediction')
                await self.send(json.dumps({'value': predict[0]}))
                await sleep(60)



