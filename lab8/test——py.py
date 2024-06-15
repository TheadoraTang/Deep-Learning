# def decode(self, dec_y, enc_hidden, state):
#     '''
#     dec_y --> (B, S), where B = batch_size, S = sequence length
#     enc_hidden --> (B, S, H), where B = batch_size, S = sequence length, H = hidden_size
#     state --> (1, B, H), where B = batch_size, H = hidden_size
#     请用RNN+attention补全解码器，其中打分函数使用点积模型
#     sent_outputs的大小应为(B, S, O) where O = num_outputs, state的大小应为(1, B, H)
#     B=64,S=8,H=100
#     '''
#     dec_emb = self.emb_layer(dec_y)
#     print("dec_y:")
#     print(dec_y.shape)
#
#     print("dec_emb:")
#     print(dec_emb.shape)
#
#     # 计算注意力权重
#     attn_weights = torch.bmm(enc_hidden, dec_emb.transpose(1, 2))  # (B, S, S)
#
#     # 对注意力权重进行softmax操作
#     attn_weights = self.softmax(attn_weights)
#
#     # 根据注意力权重对编码器隐藏状态进行加权求和
#     attn_applied = torch.bmm(attn_weights, enc_hidden)  # (B, S, H)
#
#     # # 将解码器输入与注意力加权的编码器隐藏状态拼接起来
#     # rnn_input = torch.cat((dec_emb, attn_applied), dim=2)  # (B, S, 2*H)
#     #
#     # print("rnn_input")
#     # print(rnn_input.shape)
#     print("attn_applied:")
#     print(attn_applied.shape)
#     print("state:")
#     print(state.shape)
#
#     # 使用解码器RNN进行解码
#     dec_output, state = self.decoder(attn_applied, state)  # dec_output: (B, S, H), state: (1, B, H)
#
#     print("dec_output:")
#     print(dec_output.shape)
#     # 将解码器输出经过线性层得到最终的输出
#     sent_outputs = self.linear(dec_output)  # (B, S, O)
#
#     return sent_outputs, state
#
    # def decode(self, dec_y, enc_hidden, state):
    #     '''
    #     dec_y --> (B, S), where B = batch_size, S = sequence length
    #     enc_hidden --> (B, S, H), where B = batch_size, S = sequence length, H = hidden_size
    #     state --> (1, B, H), where B = batch_size, H = hidden_size
    #     '''
    #
    #     dec_emb = self.emb_layer(dec_y)
    #
    #     # Get sequence length and batch size
    #     seq_len = dec_y.size(1)
    #     batch_size = dec_y.size(0)
    #
    #     # Initialize list to store decoder outputs
    #     sent_outputs = []
    #
    #     # Iterate through each time step
    #     for t in range(seq_len):
    #         # Compute attention scores
    #         attention_scores = torch.bmm(state.permute(1, 0, 2), enc_hidden.transpose(1, 2))  # (B, 1, S)
    #         attention_weights = torch.softmax(attention_scores, dim=2)  # (B, 1, S)
    #
    #         # Compute context vector
    #         context = torch.bmm(attention_weights, enc_hidden)  # (B, 1, H)
    #
    #         # Concatenate context vector with decoder input
    #         dec_input = torch.cat((dec_emb[:, t, :].unsqueeze(1), context), dim=2)
    #
    #         # Forward pass through decoder RNN
    #         output_t, state = self.decoder(dec_input, state)
    #
    #         # Append decoder output to list
    #         sent_outputs.append(output_t)
    #
    #     # Concatenate decoder outputs along sequence dimension
    #     sent_outputs = torch.cat(sent_outputs, dim=1)
    #
    #     # Apply linear layer to obtain final output
    #     output = self.linear(sent_outputs)
    #
    #     return output, state
#     print("dec_y的形状：", dec_y.shape)
#     dec_emb = self.emb_layer(dec_y)
#     print("enc_hidden 的形状:", enc_hidden.shape)
#     print("state 的形状:", state.shape)
#
#     attention_scores = torch.bmm(enc_hidden, state.permute(1, 2, 0))
#     print("attention_scores 的形状:", attention_scores.shape)
#
#     # 使用点积注意力机制
#     attention_weights = self.softmax(attention_scores)
#
#     context_vector = torch.bmm(enc_hidden.permute(0, 2, 1), attention_weights)
#
#     print("dec_emb的形状:", dec_emb.shape)
#     print("context_vector的形状：", context_vector.shape)
#
#     dec_emb = dec_emb.transpose(1, 2)
#     dec_input_with_context = torch.cat((dec_emb, context_vector), dim=2)
#
#     print(dec_input_with_context.shape)
#
#     # 解码器RNN
#     dec_hidden, state = self.decoder(dec_input_with_context.transpose(1, 2), state)
#     print("dec_hidden 的形状:", dec_hidden.shape)
#     print("state 的形状:", state.shape)
#     print(context_vector.shape)
#
#     # 将解码器隐藏状态和注意力上下文向量拼接起来，用于最终输出预测
#     combined_output = torch.cat((dec_hidden.transpose(1, 2), context_vector), dim=2)
#     print("combined_output", combined_output.shape)
#
#     sent_outputs = self.linear(combined_output)
#     print("send_outputs", sent_outputs)
#
#     return sent_outputs, state

import torch
import torch.nn as nn
import torch.nn.functional as F

rnn=nn.RNN(64,4,3) #input_size,hidden_size(隐藏层神经元数),num_layers(隐藏层层数)

input=torch.randn(150,3,64) #sequence_length,batch_size,input_size
h0=torch.randn(3,3,4) #num_layers*num_directions,batch_size,hidden_size

output,hn=rnn(input,h0)
print(output.shape) #sequence_length,batch_size,hidden_size -> (150,3,4)
print(hn.shape) #the same as h0 -> (3,3,4)

class Attn(nn.Module):
    def __init__(self, query_size, key_size, value_size1, value_size2, output_size):
        super(Attn, self).__init__()
        self.query_size = query_size
        self.key_size = key_size
        self.value_size1 = value_size1
        self.value_size2 = value_size2
        self.output_size = output_size

        # 线性变换层，用于将查询和键映射到相同的维度
        self.query_linear = nn.Linear(query_size, key_size)

        # 初始化注意力机制实现中第三步的线性层
        self.attn_combine = nn.Linear(query_size + value_size2, output_size)

    def forward(self, Q, K, V):
        # 将查询向量映射到与键向量相同的维度
        Q_mapped = self.query_linear(Q)

        # 计算注意力权重（点积注意力）
        attn_weights = F.softmax(torch.bmm(Q_mapped, K.transpose(1, 2)), dim=2)

        # 将注意力权重和值进行加权求和
        attn_applied = torch.bmm(attn_weights, V)

        # 将注意力权重和查询向量进行拼接
        output = torch.cat((Q, attn_applied), dim=2)

        # 将拼接后的结果进行线性变换得到最终输出
        output = self.attn_combine(output)
        return output, attn_weights


# 使用点积模型作为打分函数
query_size = 32
key_size = 32
value_size1 = 32
value_size2 = 64
output_size = 64

attn = Attn(query_size, key_size, value_size1, value_size2, output_size)
Q = torch.randn(1, 1, 32)
K = torch.randn(1, 32, 32)  # 注意键的形状与点积模型相匹配
V = torch.randn(1, 32, 64)
output = attn(Q, K, V)
print(output[0])
print(output[0].size())  # 对应output的形状 -> (1,1,64)
print(output[1])
print(output[1].size())  # 对应attn_weights的形状 -> (1,1,32)


import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, enc_units, batch_first=True)

    def forward(self, x, hidden):
        x = self.embedding(x)
        output, state = self.lstm(x, hidden)
        return output, state

    def initialize_hidden_state(self):
        return (torch.zeros(1, self.batch_sz, self.enc_units), torch.zeros(1, self.batch_sz, self.enc_units))

class BahdanauAttention(nn.Module):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = nn.Linear(units, units)
        self.W2 = nn.Linear(units, units)
        self.V = nn.Linear(units, 1)

    def forward(self, query, values):
        query_with_time_axis = query.unsqueeze(1)
        score = self.V(torch.tanh(self.W1(query_with_time_axis) + self.W2(values)))
        attention_weights = torch.softmax(score, dim=1)
        context_vector = attention_weights * values
        context_vector = torch.sum(context_vector, dim=1)
        return context_vector, attention_weights

class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim + dec_units, dec_units, batch_first=True)
        self.fc = nn.Linear(dec_units, vocab_size)
        self.attention = BahdanauAttention(dec_units)

    def forward(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden[0], enc_output)
        x = self.embedding(x)
        x = torch.cat((context_vector.unsqueeze(1), x), dim=-1)
        output, state = self.lstm(x)
        output = output.contiguous().view(-1, output.shape[2])
        x = self.fc(output)
        return x, state, attention_weights



    def decode(self, dec_y, enc_hidden, state):
        '''
        dec_y --> (B, S), 其中 B 为批量大小，S 为序列长度
        enc_hidden --> (B, S, H), 其中 B 为批量大小，S 为序列长度，H 为隐藏状态大小
        state --> (1, B, H), 其中 B 为批量大小，H 为隐藏状态大小
        请用 RNN + 注意力 机制补全解码器，其中打分函数使用点积模型
        sent_outputs 的大小应为 (B, S, O)，其中 O 为输出的维度，state 的大小应为 (1, B, H)
        '''

        dec_emb = self.emb_layer(dec_y)
        print('dec_y shape:', dec_y.shape)
        print('dec_emb shape:', dec_emb.shape)

        # 使用点积计算注意力分数
        attn_scores = torch.bmm(enc_hidden, state.permute(1, 2, 0))  # (B, S, 1)
        print('attn_scores:', attn_scores.shape)

        # 对注意力分数应用 softmax 得到注意力权重
        attn_weights = self.softmax(attn_scores)  # (B, S, 1)
        print('attn_weights:', attn_weights.shape)

        # 计算上下文向量
        context_vector = torch.bmm(attn_weights.permute(0, 2, 1), enc_hidden)  # (B, 1, H)
        print('context_vector shape:', context_vector.shape)

        # 将上下文向量与解码器隐藏状态拼接
        concat_input = torch.cat((context_vector, dec_emb), dim=1)


        # 通过解码器 RNN
        dec_output, state = self.decoder(concat_input, state)

        print('dec_output shape:', dec_output.shape)
        print('state shape:', state.shape)

        # 通过线性层得到输出logits
        sent_outputs = self.linear(torch.cat((dec_output, context_vector.expand(-1, dec_y.size(1), -1)), dim=2))

        return sent_outputs, state