# -*- coding: utf-8 -*-
# @Time    : 2019/7/3 9:05
# @Author  : hejipei
# @File    : keras_sentiment.py
""" """
from keras import Sequential
from keras.src.callbacks import EarlyStopping
from keras.src.datasets import imdb
from keras.src.layers import Embedding, Dropout, Dense, LSTM, Bidirectional, Flatten, SimpleRNN, GRU, \
    GlobalAveragePooling1D
from keras_preprocessing import sequence


'''好的博客和github'''
# https://github.com/ShawnyXiao/TextClassification-Keras/tree/master/model
# http://www.tensorflownews.com/2018/05/10/keras_gru/
# https://my.oschina.net/u/3800567/blog/2965731
# http://www.voidcn.com/article/p-alhbnusv-bon.html
# https://blog.csdn.net/shu15121856/article/category/8840507
import numpy as np


def input_data():
    print('Loading data...')
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')
    print('Average train sequence length: {}'.format(np.mean(list(map(len, x_train)), dtype=int)))
    print('Average test sequence length: {}'.format(np.mean(list(map(len, x_test)), dtype=int)))
    print('Pad sequences(samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    return x_train, y_train, x_test, y_test


def LSTM_model():
    print('Build LSTM model...')
    model = Sequential()
    model.add(Embedding(max_features, embed_size, input_length=maxlen))  # 只能作为模型的第一层 2.5w行的句子，每个词变成128维度的词向量,每个句子80个词
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def BILSTM_model():  # 双向
    print('Build BILSTM model...')
    model = Sequential()
    model.add(Embedding(max_features, embed_size, input_length=maxlen))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(32, return_sequences=True), merge_mode='concat'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def RNN_model():
    print('Build RNN model...')
    model = Sequential()
    model.add(Embedding(max_features, embed_size, input_length=maxlen))
    model.add(Dropout(0.5))
    model.add(SimpleRNN(16))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def DRNN_model():  # 双向
    print('Build DBRNN_ model...')
    model = Sequential()
    model.add(Embedding(max_features, embed_size, input_length=maxlen))
    model.add(Dropout(0.5))
    model.add(Bidirectional(SimpleRNN(16, return_sequences=True), merge_mode='concat'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def DBRNN_model():  # 组合
    print('Build DBRNN_ model...')
    model = Sequential()
    model.add(Embedding(max_features, embed_size, input_length=maxlen))
    model.add(Dropout(0.5))
    model.add(Bidirectional(SimpleRNN(16, return_sequences=True), merge_mode='concat'))
    model.add(SimpleRNN(8))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def GRU_model():
    print('Build GRU model...')
    model = Sequential()
    model.add(Embedding(max_features, embed_size, input_length=maxlen))
    model.add(Dropout(0.2))
    model.add(GRU(32))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def BIGRU_model():  # 双向
    print('Build BIGRU model...')
    model = Sequential()
    model.add(Embedding(max_features, embed_size, input_length=maxlen))
    model.add(Dropout(0.5))
    model.add(Bidirectional(GRU(32, return_sequences=True), merge_mode='concat'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# 构建模型
def Fast_text_model():
    print('Build Fast_text model...')
    model = Sequential()
    model.add(Embedding(max_features, embed_size, input_length=maxlen))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def fit_evaluate(model, x_train, y_train, x_test, y_test):
    early_stopping = EarlyStopping(monitor='val_acc', patience=5)  # 增加 EarlyStopping
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              callbacks=[early_stopping],
              validation_data=(x_test, y_test))
    score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)


if __name__ == "__main__":
    '''
    1.input_data :导入预处理好的数据
    2.xxx_model:构建模型并编译好
    3.fit_evaluate：训练并评估模型的预测accuracy值
    '''

    max_features = 25000  # 词汇表大小
    maxlen = 400  # 序列最大长度
    batch_size = 32  # 批数据量大小
    embed_size = 50  # 词向量维度
    epochs = 10  # 迭代轮次

    x_train, y_train, x_test, y_test = input_data()

    model = Fast_text_model()
    fit_evaluate(model, x_train, y_train, x_test, y_test)
    # accuracy: 0.889

    # -----------------------------------
    model = LSTM_model()
    fit_evaluate(model, x_train, y_train, x_test, y_test)
    # accuracy: 0.868
    # -----------------------------------
    model = BILSTM_model()
    fit_evaluate(model, x_train, y_train, x_test, y_test)
    # accuracy: 0.88936
    # -----------------------------------
    model = RNN_model()
    fit_evaluate(model, x_train, y_train, x_test, y_test)
    # accuracy: 0.65
    # -----------------------------------
    model = DRNN_model()
    fit_evaluate(model, x_train, y_train, x_test, y_test)
    # accuracy: 0.8718
    # -----------------------------------
    model = DBRNN_model()
    fit_evaluate(model, x_train, y_train, x_test, y_test)
    # accuracy: 0.75036
    # -----------------------------------
    model = GRU_model()
    fit_evaluate(model, x_train, y_train, x_test, y_test)
    # accuracy: 0.86364
    # -----------------------------------
    model = BIGRU_model()
    fit_evaluate(model, x_train, y_train, x_test, y_test)
    # accuracy: 0.8748
    # -----------------------------------

    model.summary()
#

#
# from keras.preprocessing import sequence
# from keras.models import Sequential
# from keras.layers import Dense,Embedding
# from keras.layers import LSTM
# from keras.datasets import imdb
#
# max_features = 20000
# maxlen = 80
# batch_size = 32
#
# print('Loading data...')
# (x_train,y_train),(x_test,y_test) = imdb.load_data(num_words= max_features )
# print(len(x_train),'train sequences')
# print(len(x_test),'test sequences')
# print('Pad sequences(samples x time)')
# x_train = sequence .pad_sequences(x_train ,maxlen= maxlen )
# x_test = sequence .pad_sequences(x_test ,maxlen= maxlen )
#
# print('x_train shape:',x_train .shape )
# print('x_test shape:',x_test .shape )
#
# print('Build model...')
# model = Sequential()
# model.add(Embedding (max_features ,128))#嵌入层将正整数下标转换为固定大小的向量。只能作为模型的第一层
# model.add(LSTM (128,dropout= 0.2,recurrent_dropout= 0.2))
# model.add(Dense(1,activation= 'sigmoid'))
# model.compile(loss= 'binary_crossentropy',optimizer= 'adam',metrics= ['accuracy'])
#
# print('Train...')
#
# model.fit(x_train ,y_train ,batch_size= batch_size ,epochs= 5,validation_data= (x_test ,y_test ))
#
# score,acc = model.evaluate(x_test ,y_test ,batch_size= batch_size )
# print('Test score:',score)
# print('Test accuracy:', acc)