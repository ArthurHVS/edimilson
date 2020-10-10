#!/usr/bin/env python
# coding: utf-8

# Código criado em 7 de Outubro de 2020
# Por: github.com/ArthurHVS

# Bloco de imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import pandas as pd
import json as json

from alpha_vantage.timeseries import TimeSeries
from sklearn.preprocessing import MinMaxScaler

from keras import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Popula os vetores, os inverte e cria um dataset com os valores.
close_array = []
date_array = []

print("Vamos treinar com a lista stock_list.json...")
with open('stock_list.json', 'r') as listFile:
    data = listFile.read()

obj = json.loads(data)

for v in obj["symbols"]:
    print(v)
    #print("Vamos construir o modelo de " + v['symbol'])
    chave = "18ETRMNXVZ4FU6SQ"
    ts = TimeSeries(key=chave, output_format="pandas", indexing_type="date")
    mainFrame, metadata = ts.get_daily_adjusted(symbol='usim5' + ".sao", outputsize="full")
    train_set = mainFrame["4. close"][:400].values
    test_set = mainFrame[400:].values

    # Define o formato dos vetores de treino e teste
    # Escalona entre 0 e 1
    sc = MinMaxScaler()
    train_set = train_set.reshape(-1, 1)
    test_set = test_set.reshape(-1, 1)
    train_set_scaled = sc.fit_transform(train_set)

    # Popula os vetores x e y dos treinos, com os valores escalonados
    x_train = []
    y_train = []
    for i in range(60, 400):
        x_train.append(train_set_scaled[i - 60: i, 0])
        y_train.append(train_set_scaled[i, 0])

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


    # Layers da rede neural. 50 neurônios. Dropout 20% (impede overfitting).
    reg = Sequential()

    reg.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    reg.add(Dropout(0.2))

    reg.add(LSTM(units=50, return_sequences=True))
    reg.add(Dropout(0.2))

    reg.add(LSTM(units=50, return_sequences=True))
    reg.add(Dropout(0.2))

    reg.add(LSTM(units=50))
    reg.add(Dropout(0.2))
    # Layer output
    reg.add(Dense(units=1))

    #compilação e fit do modelo. Por fim, salva o arquivo.
    reg.compile(optimizer="adam", loss="mean_squared_error")
    reg.fit(x_train, y_train, epochs=20, batch_size=1, verbose=2)
    reg.save('savedModels/'+'usim5')

    inp = mainFrame["4. close"][len(mainFrame) - len(test_set) - 60:].values
    inp = inp.reshape(-1, 1)
    inp = sc.transform(inp)

    x_test = []
    for i in range(60, 400):
        x_test.append(inp[i - 60: i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    pred = reg.predict(x_test)
    pred = sc.inverse_transform(pred)

    predFrame = pd.DataFrame(data=pred, index=mainFrame["4. close"][60:400].index, columns=["4. close"])
    plt.plot_date(mainFrame["4. close"][60:400].index, mainFrame["4. close"][60:400].values, fmt='-', color="red")
    plt.plot(predFrame["4. close"], color="green")

    green_patch = mpatches.Patch(color='green', label='Fechamento estimado')
    plt.legend(handles=[green_patch])

    plt.title("Resultado do treino do ativo " + "usim5")
    plt.figure(figsize=(20,11.25))
    plt.show()
    #plt.savefig(v["symbol"]+'.png')
