import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

hidden = [1024, 256, 128, 3]
batch_size = 100
num_labels=3
x = tf.placeholder(tf.float32,shape=[None,4])
y = tf.placeholder(tf.float32,shape=[None,3])
#
# l1 = tf.layers.dense(x,125,activation=tf.nn.relu)
# tmp = tf.layers.dense(l1,125*2,activation=tf.nn.relu)
# l2 = tf.layers.dense(tmp,3)
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=l2, labels=y))
# optimizer = tf.train.AdamOptimizer().minimize(loss)
# print("loss",optimizer)
# exit()


def readcsv(fileName):
    df = pd.read_csv(fileName, sep=',',usecols=range(0,5))
    cols=df.columns
    trans=df.copy()
    for i in cols:
        trans[i]=df[i].astype('category')
        trans[i] = trans[i].cat.codes
    df_train, df_test = train_test_split(trans, test_size=0.2)
    trans.to_csv('train_temp.csv', index=False)
    npmatrix=trans.as_matrix()
    df_train=df_train.as_matrix()
    df_test=df_test.as_matrix()
    return npmatrix,df_train,df_test

def neural_network_model(data):
    layers=[]
    cols =data.shape[1]
    for n_nodes in hidden:
        d = {
            "weights": tf.Variable(tf.random_normal([cols, n_nodes])),
            "biases" : tf.Variable(tf.random_normal([n_nodes]))
            }
        cols = n_nodes
        layers.append(d)

        output_layer = {
        "weights" : tf.Variable(tf.random_normal([hidden[-1], num_labels])),
        "biases" : tf.Variable(tf.random_normal([num_labels]))
                        }
    input_tensor =x
    for layer in layers:
        v = tf.add(tf.matmul(input_tensor, layer["weights"]), layer["biases"])
        v = tf.nn.relu(v)
        input_tensor = v

    output = tf.add(tf.matmul(input_tensor, output_layer["weights"]), output_layer["biases"])
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    return output

def train_neural_network(trainxdata, ckpt_file=None, save=True):
    prediction = neural_network_model(trainxdata)
    # print prediction
    # exit()
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    epochs = 4000
    arr = []
    # plt.ion();


    with tf.Session() as sess:
        saver = tf.train.Saver()

        if ckpt_file:
            print "Restoring saved model from : {}".format(ckpt_file)
            saver.restore(sess, ckpt_file)
        else:
            sess.run(tf.global_variables_initializer())
            for epoch in range(epochs):
                epoch_loss = 0
                idx = 0
                # while idx < trainxdata.shape[0]:
                _, c = sess.run([optimizer, cost], feed_dict= {
                    x : trainxdata,
                    y : trainydata
                    })
                epoch_loss += c
                # idx += batch_size
                print "Epoch : {} of {}, loss : {}".format(epoch + 1, epochs, epoch_loss)
                    # plt.scatter(epoch, epoch_loss)
                    # plt.pause(0.001)
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, "float"))
            print "Accuracytrain : {}".format(accuracy.eval({
                x : trainxdata,
                y : trainydata
            }))
            print "Accuracy : {}".format(accuracy.eval({
                x : testxdata,
                y : testydata
            }))
            # if save:
            #     saver.save(sess=sess, save_path="./bot.ckpt")
        tt = tf.argmax(prediction, 1)
        res = tt.eval(feed_dict = {
            x : testxdata,
            y : testydata
        })




def trnasforming_data(npmatrix):
    npmatrix=(npmatrix-npmatrix.mean(axis=0))/npmatrix.std(axis=0)
    #np.nan_to_num(npmatrix)
    # trans=tf.convert_to_tensor(npmatrix,dtype=tf.float32)
    # trans=trans-tf.reduce_mean(trans,axis=0)
    return npmatrix

data,data_train,data_test=readcsv("temp.csv")
trainxdata=trnasforming_data(data_train[:,:-1])
testxdata=trnasforming_data(data_test[:,:-1])
testxdata=testxdata.astype(np.float32)
trainxdata=trainxdata.astype(np.float32)
trainydata = to_categorical(data_train[:,-1])
testydata = to_categorical(data_test[:,-1])
# print(trainxdata.shape,trainydata.shape,testxdata.shape,testydata.shape)
# exit()
train_neural_network(trainxdata)
