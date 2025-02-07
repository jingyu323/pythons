import re

import numpy as np
import pandas as pd
from keras import Input, Model
from keras.src.layers import Embedding, Bidirectional, TimeDistributed, Dense, LSTM
from keras_core.src.utils import to_categorical


def demo():
    text = open('msr_train_10.txt', encoding='gbk').read()
    text = text.split('\n')
    print(len(text))

    # {B:begin, M:middle, E:end, S:single}，分别代表每个状态代表的是该字在词语中的位置，
    # B代表该字是词语中的起始字，M代表是词语中的中间字，E代表是词语中的结束字，S则代表是单字成词

    print(text)

    # 设置参数
    # 词向量长度
    word_size = 128
    # 设置最长的一句话为32个字
    maxlen = 32
    # 批次大小
    batch_size = 1024

    # 根据符号分句
    text = u''.join(text)
    text = re.split(u'[，。！？、]/[bems]', text)
    print("text=====", text)

    # 训练集数据
    data = []
    # 标签
    label = []

    # 得到所有的数据和标签
    def get_data(s):
        s = re.findall('(.)/(.)', s)
        print("s===", s)
        if s:
            s = np.array(s)
            # 返回数据和标签，0为数据，1为标签
            return list(s[:, 0]), list(s[:, 1])

    for s in text:
        d = get_data(s)
        if d:
            data.append(d[0])
            label.append(d[1])

    test = re.findall('(.)/(.)', '你/s  只/b  有/e  把/s  事/b  情/e  做/b  好/e')
    print("test12========", test)

    # 定义一个dataframe存放数据和标签
    d = pd.DataFrame(index=range(len(data)))
    d['data'] = data
    d['label'] = label
    # 提取data长度小于等于maxlen的数据
    d = d[d['data'].apply(len) <= maxlen]
    # 重新排列index
    d.index = range(len(d))

    print(range(len(d)))
    print("d.index === ", d)

    chars = []
    for i in data:
        chars.extend(i)

    chars = pd.Series(chars).value_counts()

    print("========================")
    chars[:] = range(1, len(chars) + 1)
    print(chars)

    tag = pd.Series({'s': 0, 'b': 1, 'm': 2, 'e': 3, 'x': 4})

    # # 把中文变成编号，再补0
    # d['x'] = d['data'].apply(lambda x: np.array(list(chars[x])+[0]*(maxlen-len(x))))
    # # 把标签变成编号，再补0
    # d['y'] = d['label'].apply(lambda x: np.array(list(map(lambda y:np_utils.to_categorical(y,5), tag[x].reshape((-1,1))))+[np.array([[0,0,0,0,1]])]*(maxlen-len(x))))

    def data_helper(x):
        x = list(chars[x]) + [0] * (maxlen - len(x))
        return np.array(x)

    def label_helper(x):
        x = list(map(lambda y: to_categorical(y, 5), tag[x].values.reshape((-1, 1))))
        x = x + [np.array([[0, 0, 0, 0, 1]])] * (maxlen - len(x))
        return np.array(x)

    d['x'] = d['data'].apply(data_helper)
    d['y'] = d['label'].apply(label_helper)
    print(d['data'][0])
    print(d['x'][0])
    print(d['label'][0])
    print(d)

    sequence = Input(shape=(maxlen,), dtype='int32')
    # 词汇数，词向量长度，输入的序列长度，是否忽略0值
    embedded = Embedding(len(chars) + 1, word_size, input_length=maxlen, mask_zero=True)(sequence)
    # 双向RNN包装器
    blstm = Bidirectional(LSTM(64, return_sequences=True), merge_mode='sum')(embedded)
    # 该包装器可以把一个层应用到输入的每一个时间步上
    output = TimeDistributed(Dense(5, activation='softmax'))(blstm)
    # 定义模型输出输出
    model = Model(inputs=sequence, outputs=output)
    # 定义代价函数，优化器
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'], run_eagerly=True)
    model.summary()
    # print(np.array(list(d['x'])).shape)
    # print(np.array(list(d['y'])).reshape((-1,maxlen,5)).shape)
    model.fit(np.array(list(d['x'])), np.array(list(d['y'])).reshape((-1, maxlen, 5)), batch_size=batch_size, epochs=20)
    # model.save('seq2seq.h5')


if __name__ == '__main__':
    demo()
