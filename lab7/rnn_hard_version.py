# import torch
# from torch import nn
# import numpy as np
#
# np.random.seed(2022)
# torch.manual_seed(2022)
#
# class GRU(nn.Module):
#     def __init__(self, num_inputs, num_hiddens):
#         super(GRU, self).__init__()
#         # 隐藏层参数
#
#         self.W_xz = torch.nn.Parameter(torch.tensor(np.random.normal(0, 0.01, size=(num_inputs, num_hiddens)), dtype=torch.float32))
#         self.W_hz = torch.nn.Parameter(torch.tensor(np.random.normal(0, 0.01, size=(num_hiddens, num_hiddens)), dtype=torch.float32))
#         self.b_z = torch.nn.Parameter(torch.zeros(num_hiddens))
#
#         self.W_xr = torch.nn.Parameter(torch.tensor(np.random.normal(0, 0.01, size=(num_inputs, num_hiddens)), dtype=torch.float32))
#         self.W_hr = torch.nn.Parameter(torch.tensor(np.random.normal(0, 0.01, size=(num_hiddens, num_hiddens)), dtype=torch.float32))
#         self.b_r = torch.nn.Parameter(torch.zeros(num_hiddens))
#
#         self.W_xh = torch.nn.Parameter(torch.tensor(np.random.normal(0, 0.01, size=(num_inputs, num_hiddens)), dtype=torch.float32))
#         self.W_hh = torch.nn.Parameter(torch.tensor(np.random.normal(0, 0.01, size=(num_hiddens, num_hiddens)), dtype=torch.float32))
#         self.b_h = torch.nn.Parameter(torch.zeros(num_hiddens))
#
#     def forward(self, inputs, H):
#         '''
#         补全GRU的前向传播，
#         不能调用pytorch中内置的GRU函数及操作
#         '''
#         # ==========
#         # todo '''请补全GRU网络前向传播'''
#         # ==========
#         return outputs, H
#
#
# class Sequence_Modeling(nn.Module):
#     def __init__(self, vocab_size, embedding_size, num_outputs, hidden_size):
#         super(Sequence_Modeling, self).__init__()
#         self.emb_layer = nn.Embedding(vocab_size, embedding_size)
#         self.gru_layer = GRU(embedding_size, hidden_size)
#         self.linear = nn.Linear(hidden_size, num_outputs)
#
#     def forward(self, sent, state):
#         '''
#         sent --> (B, S) where B = batch size, S = sequence length
#         sent_emb --> (B, S, I) where B = batch size, S = sequence length, I = num_inputs
#         state --> (1, B, H), where B = batch_size, num_hiddens
#         你需要利用定义好的emb_layer, gru_layer和linear，
#         补全代码实现歌词预测功能，
#         sent_outputs的大小应为(B, S, O) where O = num_outputs, state的大小应为(1, B, H)
#         '''
#
#         # ==========
#         # todo '''请补全代码'''
#         # ==========
#         return sent_outputs, state
#
#
# if __name__ == '__main__':
#     model = Sequence_Modeling()

import torch
from torch import nn
import numpy as np

np.random.seed(2022)
torch.manual_seed(2022)

class GRU(nn.Module):
    def __init__(self, num_inputs, num_hiddens):
        super(GRU, self).__init__()
        # 隐藏层参数

        self.W_xz = torch.nn.Parameter(torch.tensor(np.random.normal(0, 0.01, size=(num_inputs, num_hiddens)), dtype=torch.float32))
        self.W_hz = torch.nn.Parameter(torch.tensor(np.random.normal(0, 0.01, size=(num_hiddens, num_hiddens)), dtype=torch.float32))
        self.b_z = torch.nn.Parameter(torch.zeros(num_hiddens))

        self.W_xr = torch.nn.Parameter(torch.tensor(np.random.normal(0, 0.01, size=(num_inputs, num_hiddens)), dtype=torch.float32))
        self.W_hr = torch.nn.Parameter(torch.tensor(np.random.normal(0, 0.01, size=(num_hiddens, num_hiddens)), dtype=torch.float32))
        self.b_r = torch.nn.Parameter(torch.zeros(num_hiddens))

        self.W_xh = torch.nn.Parameter(torch.tensor(np.random.normal(0, 0.01, size=(num_inputs, num_hiddens)), dtype=torch.float32))
        self.W_hh = torch.nn.Parameter(torch.tensor(np.random.normal(0, 0.01, size=(num_hiddens, num_hiddens)), dtype=torch.float32))
        self.b_h = torch.nn.Parameter(torch.zeros(num_hiddens))

    def forward(self, inputs, H):
        '''
        补全GRU的前向传播，
        不能调用pytorch中内置的GRU函数及操作
        '''
        Z = torch.sigmoid(torch.matmul(inputs, self.W_xz) + torch.matmul(H, self.W_hz) + self.b_z)
        R = torch.sigmoid(torch.matmul(inputs, self.W_xr) + torch.matmul(H, self.W_hr) + self.b_r)
        H_tilde = torch.tanh(torch.matmul(inputs, self.W_xh) + torch.matmul(R * H, self.W_hh) + self.b_h)
        H_new = Z * H + (1 - Z) * H_tilde

        return H_new


class Sequence_Modeling(nn.Module):
    def __init__(self, vocab_size, embedding_size, num_outputs, hidden_size):
        super(Sequence_Modeling, self).__init__()
        self.emb_layer = nn.Embedding(vocab_size, embedding_size)
        self.gru_layer = GRU(embedding_size, hidden_size)
        self.linear = nn.Linear(hidden_size, num_outputs)

    def forward(self, sent, state):
        sent_emb = self.emb_layer(sent)  # (B, S, I)

        sent_outputs = []

        for i in range(sent_emb.size(1)):
            input_emb = sent_emb[:, i, :]  # (B, I)
            state = self.gru_layer(input_emb, state.squeeze(0))
            output = self.linear(state)
            sent_outputs.append(output.unsqueeze(1))

        sent_outputs = torch.cat(sent_outputs, dim=1)  # (B, S, O)

        state = state.unsqueeze(0).detach()  # (1, B, H)

        return sent_outputs, state

