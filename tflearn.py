import tensorflow as tf
import tflearn
import numpy as np

from tflearn.data_utils import load_csv

#target_column : classification value column
#n_classes : number of classifications
from auditng.datasets import processed
processed.download_dataset('processed_dataset.csv')

data, labels = load_csv("processed_dataset.csv", target_column=0, categorical_labels=True, n_classes=2)
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

