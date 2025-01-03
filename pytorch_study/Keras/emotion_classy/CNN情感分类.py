# coding: utf-8

import pandas as pd 
import numpy as np 
import jieba # pip install jieba
from keras import Input, Model
from keras.src.layers import Embedding, Flatten, concatenate, Dense, MaxPooling1D, Conv1D, Dropout
from keras.src.legacy.preprocessing.text import Tokenizer
from keras.src.utils import pad_sequences


# 读入数据
neg=pd.read_excel('data/neg.xls',header=None).astype('str')
pos=pd.read_excel('data/pos.xls',header=None).astype('str')


#合并语料
pn = pd.concat([pos,neg],ignore_index=True) 
#计算语料数目
neglen = len(neg)
print(neglen)
poslen = len(pos) 
print(poslen)


#定义分词函数
cw = lambda x: list(jieba.cut(x))
pn['words'] = pn[0].apply(cw)

# 一行数据最多的词汇数
max_document_length = max([len(x) for x in pn['words']])

print(max_document_length)
# 设置一个评论最多1000个词
max_document_length = 1000

texts = [' '.join(x) for x in pn['words']]
# 查看一条评论
print(texts[-2])



# 实例化分词器，设置字典中最大词汇数为30000
tokenizer = Tokenizer(num_words=30000)
# 传入我们的训练数据，建立词典
tokenizer.fit_on_texts(texts) 
# 把词转换为编号，词的编号根据词频设定，频率越大，编号越小
sequences = tokenizer.texts_to_sequences(texts) 
# 把序列设定为1000的长度，超过1000的部分舍弃，不到1000则补0
sequences = pad_sequences(sequences, maxlen=1000, padding='post')  
sequences = np.array(sequences)


# 词对应编号的字典
dict_text = tokenizer.word_index
print(dict_text['也'])





# 定义标签
positive_labels = [[0, 1] for _ in range(poslen)]
negative_labels = [[1, 0] for _ in range(neglen)]
y = np.concatenate([positive_labels, negative_labels], 0)

# 打乱数据
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = sequences[shuffle_indices]
y_shuffled = y[shuffle_indices]

# 数据集切分为两部分
test_sample_index = -1 * int(0.1 * float(len(y)))
x_train, x_test = x_shuffled[:test_sample_index], x_shuffled[test_sample_index:]
y_train, y_test = y_shuffled[:test_sample_index], y_shuffled[test_sample_index:]


# 定义函数式模型
# 模型输入
sequence_input = Input(shape=(1000,))
# Embedding层，30000表示30000个词，每个词对应的向量为128维，序列长度为1000
embedding_layer = Embedding(30000,
                            128,
                            input_length=1000)
embedded_sequences = embedding_layer(sequence_input)

# 卷积核大小为3
cnn1 = Conv1D(filters=32, kernel_size=3, activation='relu')(embedded_sequences)
cnn1 = MaxPooling1D(pool_size=5)(cnn1)
cnn1 = Conv1D(filters=32, kernel_size=3, activation='relu')(cnn1)
cnn1 = MaxPooling1D(pool_size=5)(cnn1)
cnn1 = Conv1D(filters=32, kernel_size=3, activation='relu')(cnn1)
cnn1 = MaxPooling1D(pool_size=37)(cnn1)
cnn1 = Flatten()(cnn1)
# 卷积核大小为4
cnn2 = Conv1D(filters=32, kernel_size=4, activation='relu')(embedded_sequences)
cnn2 = MaxPooling1D(pool_size=5)(cnn2)
cnn2 = Conv1D(filters=32, kernel_size=4, activation='relu')(cnn2)
cnn2 = MaxPooling1D(pool_size=5)(cnn2)
cnn2 = Conv1D(filters=32, kernel_size=4, activation='relu')(cnn2)
cnn2 = MaxPooling1D(pool_size=36)(cnn2)
cnn2 = Flatten()(cnn2)
# 卷积核大小为5
cnn3 = Conv1D(filters=32, kernel_size=5, activation='relu')(embedded_sequences)
cnn3 = MaxPooling1D(pool_size=5)(cnn3)
cnn3 = Conv1D(filters=32, kernel_size=5, activation='relu')(cnn3)
cnn3 = MaxPooling1D(pool_size=5)(cnn3)
cnn3 = Conv1D(filters=32, kernel_size=5, activation='relu')(cnn3)
cnn3 = MaxPooling1D(pool_size=35)(cnn3)
cnn3 = Flatten()(cnn3)

# 合并
merge = concatenate([cnn1, cnn2, cnn3], axis=1)
# 全链接层
x = Dense(128, activation='relu')(merge)
# Dropout层
x = Dropout(0.5)(x)
# 输出层
preds = Dense(2, activation='softmax')(x)
# 定义模型
model = Model(sequence_input, preds)

model.summary()


# 训练模型
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

model.fit(x_train, y_train,
          batch_size=128,
          epochs=5,
          validation_data=(x_test, y_test))


# 预测
def predict(text):
    # 对句子分词
    cw = list(jieba.cut(text)) 
    word_id = []
    # 把词转换为编号
    for word in cw:
        try:
            temp = dict_text[word]
            word_id.append(temp)
        except:
            word_id.append(0)
    word_id = np.array(word_id)
    word_id = word_id[np.newaxis,:]
    sequences = pad_sequences(word_id, maxlen=1000, padding='post')  
    result = np.argmax(model.predict(sequences))
    if(result==1):
        print("positive comment")
    else:
        print("negative comment")


predict("东西质量不错")


# <h3 align = "center">欢迎大家关注我的公众号，或者加我的微信与我交流。</h3>
# <center><img src="wx.png" alt="FAO" width="300"></center> 




