import jieba
import numpy as np
from keras import Sequential
from keras.src.layers import Embedding, Bidirectional, LSTM, Dense
from keras.src.legacy.preprocessing.text import Tokenizer
from keras.src.utils import pad_sequences

# 示例文本数据
texts = ["我喜欢这部电影", "这部电影很糟糕", "那本书很不错", "我讨厌那本书"]
labels = [1, 0, 1, 0]  # 1 表示积极，0 表示消极
# 文本预处理
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

# 传入我们的训练数据，建立词典
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
dict_text = tokenizer.word_index

print("sequences=",sequences)
print("dict_text=",dict_text)
print(tokenizer.word_index)


# 词对应编号的字典
dict_text = tokenizer.word_index
print(dict_text['我喜欢这部电影'])


max_sequence_length = max([len(seq) for seq in sequences])
print("max_sequence_length=",max_sequence_length)
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
padded_sequences = np.array(padded_sequences)
labels = np.array(labels)

# 把序列设定为1000的长度，超过1000的部分舍弃，不到1000则补0

sequences = np.array(sequences)


print("padded_sequences=",padded_sequences)
print(padded_sequences.shape)

# 构建 BiLSTM 模型
model = Sequential()
# 嵌入层，将文本转换为向量表示
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=max_sequence_length))
# 双向 LSTM 层
model.add(Bidirectional(LSTM(64)))

# 全连接层，用于分类
model.add(Dense(1, activation='sigmoid'))

# 编译模型
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
model.fit(padded_sequences, labels, epochs=10, batch_size=10)

# 评估模型
loss, accuracy = model.evaluate(padded_sequences, labels)
print(f"Loss: {loss}, Accuracy: {accuracy}")


# 预测
def predict(text):
    # 对句子分词
    word_id = []
    # temp = dict_text[text]

    print("word_id=", word_id)
    word_id = np.array(word_id)

    # word_id = word_id[np.newaxis, :]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text)
    sequences = tokenizer.texts_to_sequences(text)
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
    print("padded_sequences=",padded_sequences)
    res33= model.predict(padded_sequences , verbose=1)
    print("res33=", res33)
    result = np.argmax(res33)
    print("result=",result)



predict(["那本书很不错"])