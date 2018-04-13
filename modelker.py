import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation

def readcsv(fileName):
    df = pd.read_csv(fileName, sep=',')
    cols=df.columns
    trans=df.copy()
    for i in cols:
        trans[i]=df[i].astype('category')
        trans[i] = trans[i].cat.codes
    # df_train, df_test = train_test_split(df, test_size=0.1)
    trans.to_csv('train_temp.csv', index=False)
    npmatrix=trans.as_matrix()
    return npmatrix

def trnasforming_data(npmatrix):
    npmatrix=(npmatrix-npmatrix.mean(axis=0))
    # trans=tf.convert_to_tensor(npmatrix,dtype=tf.float32)
    # trans=trans-tf.reduce_mean(trans,axis=0)
    return trans

def kerasmodel(data):
    model = Sequential()
    model.add(Dense(32, input_dim=data.shape[1]))
    model.add(Activation('relu'))

data=readcsv("temp.csv")
transdata=trnasforming_data(data)
kerasmodel(transdata)
