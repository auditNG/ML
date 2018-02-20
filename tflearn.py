import tensorflow as tf
import tflearn
import numpy as np

from tflearn.data_utils import load_csv

#target_column : classification value column
#n_classes : number of classifications
from tflearn.datasets import titanic
titanic.download_dataset('titanic_dataset.csv')

data, labels = load_csv("titanic_dataset.csv", target_column=0, categorical_labels=True, n_classes=2)
print labels
def preprocess(data, column_to_ignore):
	for id in sorted(column_to_ignore, reverse=True):
		[r.pop(id) for r in data]

	for i in range(len(data)):
		data[i][1] = 1. if data[i][1] == "female" else 0.
	return np.array(data, dtype=np.float32)


to_ignore = [1, 6]

data = preprocess(data, to_ignore)

net = tflearn.input_data(shape=[None, 6])
net = tflearn.fully_connected(net, 32) #1st hidden layer
net = tflearn.fully_connected(net, 32) #2nd hidder layer
net = tflearn.fully_connected(net, 32) #3rd hidder layer
net = tflearn.fully_connected(net, 2, activation='softmax') #output layer
net = tflearn.regression(net)

model = tflearn.DNN(net)

model.fit(data, labels, n_epoch=10, batch_size=16, show_metric=True)


jack = [3, 'Jack Dawson', "male", 19, 0, 0, 'N/A', 5.0000]
kate = [1, 'Rose DeWitt Bukater', 'female', 17, 1, 2, 'N/A', 100.0000]

jack, kate = preprocess([jack, kate], to_ignore)

pred = model.predict([jack, kate])
print "Dicaprio survival rate : " + str(pred[0][1])
print "Winslet survival rate : " + str(pred[1][1])
