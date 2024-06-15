import torch
from torch import nn


class Sequence_Modeling(nn.Module):
    def __init__(self, vocab_size, embedding_size, num_outputs, hidden_size):
        super(Sequence_Modeling, self).__init__()
        self.emb_layer = nn.Embedding(vocab_size, embedding_size)
        self.encoder = nn.RNN(embedding_size, hidden_size, batch_first=True)
        # 修改了self.mlp
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size)
        )
        self.decoder = nn.RNN(embedding_size + hidden_size, hidden_size, batch_first=True)
        self.softmax = nn.Softmax(dim=2)
        self.linear = nn.Linear(hidden_size, num_outputs)

    def encode(self, enc_x, state):
        enc_emb = self.emb_layer(enc_x)
        enc_hidden, state = self.encoder(enc_emb, state)

        return enc_hidden, state

    def decode(self, dec_y, enc_hidden, state):
        dec_embs = self.emb_layer(dec_y)
        # ==========
        '''
        dec_y --> (B, S), 其中 B 为批量大小，S 为序列长度
        enc_hidden --> (B, S, H), 其中 B 为批量大小，S 为序列长度，H 为隐藏状态大小
        state --> (1, B, H), 其中 B 为批量大小，H 为隐藏状态大小
        请用 RNN + 注意力 机制补全解码器，其中打分函数使用点积模型
        sent_outputs 的大小应为 (B, S, O)，其中 O 为输出的维度，state 的大小应为 (1, B, H)
        '''
        # todo '''请补全代码'''
        # ==========

        return sent_outputs, state


if __name__ == '__main__':
    model = Model_NP()