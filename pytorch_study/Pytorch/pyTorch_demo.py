import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import IMDB

# 加载数据
train_data, test_data = IMDB.splits(train='train', test='test')

# 构建词汇表
TEXT = data.Field(tokenize='spacy', lower=True)
LABEL = data.LabelField(dtype=torch.float)
fields = [(None, None), ('text', TEXT), ('label', LABEL)]
train_data, test_data = datasets.IMDB.splits(fields)
TEXT.build_vocab(train_data, max_size=25000)
LABEL.build_vocab(train_data)


# 构建模型
class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.rnn(embedded)
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

    # 参数设置


vocab_size = len(TEXT.vocab)
embedding_dim = 100
hidden_dim = 256
output_dim = 1
n_layers = 2
bidirectional = True
dropout = 0.5

# 实例化模型
model = RNN(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout)

# 定义损失函数和优化器
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters())

# 训练模型
for epoch in range(5):
    for batch in train_data:
        optimizer.zero_grad()
        text = batch.text
        label = batch.label
        output = model(text)
        loss = criterion(output.squeeze(1), label)
        loss.backward()
        optimizer.step()

