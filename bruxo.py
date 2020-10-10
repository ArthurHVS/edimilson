import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from alpha_vantage.timeseries import TimeSeries
from datetime import datetime, timedelta

td = timedelta(days=1)

chave = "18ETRMNXVZ4FU6SQ"
sc = MinMaxScaler()
ts = TimeSeries(key=chave, output_format="pandas", indexing_type="date")

model = tf.keras.models.load_model('./savedModels/ABEV3') 
mainFrame, metadata = ts.get_daily_adjusted("abev3.sao", outputsize="full")

inp = mainFrame["4. close"][:340].values

inp = inp.reshape(-1, 1)
inp = sc.fit_transform(inp)

x_input = []
x_index = []
for i in range(0,340):
    x_input.append(inp[i])
x_input = np.array(x_input)
x_input = np.reshape(x_input, (x_input.shape[0], x_input.shape[1], 1))

pred = model.predict(x_input)
pred = sc.inverse_transform(pred)
print()
#mainFrame["4. close"][60:400].index

#predFrame = pd.DataFrame(data=pred, index=ind, columns=["4. close"])

#plt.plot(mainFrame["4. close"][:340].values, color="red")
#plt.plot(predFrame[""].values, color="green")
#plt.show()
