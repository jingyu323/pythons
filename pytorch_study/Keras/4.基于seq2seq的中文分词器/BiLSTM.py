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

sequences = tokenizer.texts_to_sequences(texts)

print("sequences=",sequences)
print(tokenizer.word_index)



max_sequence_length = max([len(seq) for seq in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
padded_sequences = np.array(padded_sequences)



print(max_sequence_length)
print(padded_sequences)
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
model.fit(padded_sequences, labels, epochs=10, batch_size=1)

# 评估模型
loss, accuracy = model.evaluate(padded_sequences, labels)
print(f"Loss: {loss}, Accuracy: {accuracy}")

