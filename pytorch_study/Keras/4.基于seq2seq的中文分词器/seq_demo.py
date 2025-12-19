import re

import numpy as np
import pandas as pd
from keras import Input, Model
from keras.src.layers import Embedding, Bidirectional, TimeDistributed, Dense, LSTM
from keras_core.src.utils import to_categorical
from keras.src.saving import load_model


def demo():
    text = open('msr_train.txt', encoding='gbk').read()
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
    # batch_size = 512

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
    print("chars===", chars)

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
    print(np.array(list(d['x'])).shape)
    print(np.array(list(d['y'])).reshape((-1, maxlen, 5)).shape)

    print(np.array(list(d['x'])))
    print(np.array(list(d['y'])).reshape((-1, maxlen, 5)))
    print(np.array(list(d['y'])))
    model.fit(np.array(list(d['x'])), np.array(list(d['y'])).reshape((-1, maxlen, 5)), batch_size=batch_size, epochs=20)
    model.save('seq2seq.keras')


# demo()

maxlen = 32

# 使用全数据
text = open('msr_train.txt', encoding='gbk').read()
text = text.split('\n')

# 根据符号分句
text = u''.join(text)
text = re.split(u'[，。！？、]/[bems]', text)

# 训练集数据
data = []
# 标签
label = []


# 得到所有的数据和标签
def get_data(s):
    s = re.findall('(.)/(.)', s)
    if s:
        s = np.array(s)
        # 返回数据和标签，0为数据，1为标签
        return list(s[:, 0]), list(s[:, 1])


for s in text:
    d = get_data(s)
    if d:
        data.append(d[0])
        label.append(d[1])

# 定义一个dataframe存放数据和标签
d = pd.DataFrame(index=range(len(data)))
d['data'] = data
d['label'] = label
# 提取data长度小于等于maxlen的数据
d = d[d['data'].apply(len) <= maxlen]
# 重新排列index
d.index = range(len(d))

# 统计所有字，给每个字编号
chars = []
for i in data:
    chars.extend(i)

chars = pd.Series(chars).value_counts()
chars[:] = range(1, len(chars) + 1)

print("load model")
model = load_model('seq2seq.keras')

model.summary()

# 统计状态转移
dict_label = {}
for label in d['label']:
    for i in range(len(label) - 1):
        tag = label[i] + label[i + 1]
        dict_label[tag] = dict_label.get(tag, 0) + 1
print(dict_label)

# 计算状态转移总次数
sum_num = 0
for value in dict_label.values():
    sum_num = sum_num + value

print("sum_num-==", sum_num)
# 计算状态转移概率
p_ss = dict_label['ss'] / sum_num
p_sb = dict_label['sb'] / sum_num
p_bm = dict_label['bm'] / sum_num
p_be = dict_label['be'] / sum_num
p_mm = dict_label['mm'] / sum_num
p_me = dict_label['me'] / sum_num
p_es = dict_label['es'] / sum_num
p_eb = dict_label['eb'] / sum_num

# 定义状态转移矩阵
transfer = [[p_ss, p_sb, 0, 0],
            [0, 0, p_bm, p_be],
            [0, 0, p_mm, p_me],
            [p_es, p_eb, 0, 0]]

# # 定义状态转移矩阵
# transfer = [[1,1,0,0],
#             [0,0,1,1],
#             [0,0,1,1],
#             [1,1,0,0]]

# 根据符号断句
cuts = re.compile(u'([\da-zA-Z ]+)|[。，、？！\.\?,!]')


# 预测分词
def predict(sentence):
    # 如果句子大于最大长度，只取maxlen个词
    if len(sentence) > maxlen:
        sentence = sentence[:maxlen]

    print(list(sentence))
    print(chars[list(sentence)].fillna(0).astype(int))
    print(maxlen - len(sentence))
    print([0] * (maxlen - len(sentence)))

    sen_data = np.array([list(chars[list(sentence)].fillna(0).astype(int)) + [0] * (maxlen - len(sentence))])
    print(sen_data)

    # 预测结果，先把句子变成编号的形式，如果出现生僻字就填充0，然后给句子补0直到maxlen的长度。预测得到的结果只保留跟句子有效数据相同的长度
    result = \
        model.predict(np.array([list(chars[list(sentence)].fillna(0).astype(int)) + [0] * (maxlen - len(sentence))]))[
            0][
        :len(sentence)]

    print("result===", result)

    # 存放最终结果
    y = []
    # 存放临时概率值
    prob = []
    # 计算最大转移概率
    # 首先计算第1个字和第2个字,统计16种情况的概率
    # result[0][j]第1个词的标签概率
    # result[1][k]第2个词的标签概率
    # transfer[j][k]对应的转移概率矩阵的概率
    for j in range(4):
        for k in range(4):
            # 第1个词为标签j的概率*第2个词为标签k的概率*jk的转移概率
            prob.append(result[0][j] * result[1][k] * transfer[j][k])

        # 计算前一个词的的标签
        word1 = np.argmax(prob) // 4
        # 计算后一个词的标签
        word2 = np.argmax(prob) % 4
        # 保存结果
        y.append(word1)
        y.append(word2)
        # 从第2个字开始
        for i in range(1, len(sentence) - 1):
            # 存放临时概率值
            prob = []
            # 计算前一个字和后一个字的所有转移概率
            for j in range(4):
                # 前一个字的标签已知为word2
                prob.append(result[i][word2] * result[i + 1][j] * transfer[word2][j])
            # 计算后一个字的标签
            word2 = np.argmax(prob) % 4
            # 保存结果
            y.append(word2)

    # 分词
    words = []
    for i in range(len(sentence)):
        # 如果标签为s或b，append到结果的list中
        if y[i] in [0, 1]:
            words.append(sentence[i])
        else:
            # 如果标签为m或e，在list最后一个元素中追加内容
            words[-1] += sentence[i]
    return words


# 分句
def cut_word(s):
    result = []
    # 指针设置为0
    j = 0
    # 根据符号断句
    for i in cuts.finditer(s):
        # 对符号前的部分分词
        print("ddddddd====", s[j:i.start()])
        result.extend(predict(s[j:i.start()]))
        # 加入符号
        result.append(s[i.start():i.end()])
        # 移动指针到符号后面
        j = i.end()
    # 对最后的部分进行分词
    result.extend(predict(s[j:]))
    return result


if __name__ == '__main__':
    cut_word('基于seq2seq的中文分词器')
    res = cut_word('人们常说生活是一部教科书')

    print("res ====  ", res)
    #
    # cut_word('广义相对论是描写物质间引力相互作用的理论')
    #
    rssss = cut_word('我爱北京天安门，天安门上太阳升')
    res2 = model.predict(
        np.array([list(chars[list('今天天气很好')].fillna(0).astype(int)) + [0] * (maxlen - len('今天天气很好'))]))

    print(res2)
    print(rssss)
