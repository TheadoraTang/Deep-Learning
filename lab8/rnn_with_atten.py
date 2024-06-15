import torch
from torch import nn

torch.manual_seed(2023)

class Sequence_Modeling(nn.Module):
    def __init__(self, vocab_size, embedding_size, num_outputs, hidden_size):
        super(Sequence_Modeling, self).__init__()
        self.emb_layer = nn.Embedding(vocab_size, embedding_size)
        self.encoder = nn.RNN(embedding_size, hidden_size, batch_first=True)
        # 修改了self.mlp
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )
        self.decoder = nn.RNN(embedding_size + hidden_size, hidden_size, batch_first=True)
        self.softmax = nn.Softmax(dim=2)
        self.linear = nn.Linear(hidden_size, num_outputs)

    def encode(self, enc_x, state):
        enc_emb = self.emb_layer(enc_x)
        enc_hidden, state = self.encoder(enc_emb, state)
        return enc_hidden, state

    def decode(self, dec_y, enc_hidden, state):
        '''
        dec_y --> (B, S), where B = batch_size, S = sequence length
        enc_hidden --> (B, S, H), where B = batch_size, S = sequence length, H = hidden_size
        state --> (1, B, H), where B = batch_size, H = hidden_size
        请用RNN+attention补全解码器，其中打分函数使用点积模型
        sent_outputs的大小应为(B, S, O) where O = num_outputs, state的大小应为(1, B, H)
        '''
        dec_embs = self.emb_layer(dec_y)
        dec_input = torch.cat([dec_embs, enc_hidden[:, -1, :].unsqueeze(1).repeat(1, dec_embs.size(1), 1)], dim=2)
        H, state = self.decoder(dec_input, state)
        attention_scores = torch.bmm(H, enc_hidden.transpose(1, 2))
        attention_result = torch.bmm(self.softmax(attention_scores), enc_hidden)
        combined_outputs = torch.cat([H, attention_result], dim=2)
        # 确保`combined outputs、的维度符合'self.mlp`
        combined_outputs = combined_outputs[:, :, :self.mlp[0].in_features]
        # 使用'self.mlp’进行线性转换
        combined_outputs = self.mlp(combined_outputs)
        # 通过线性层计算最终输出
        sent_outputs = self.linear(combined_outputs)


        dec_emb = self.emb_layer(dec_y)

        # 计算注意力权重
        attn_weights = torch.bmm(enc_hidden, state.permute(1, 2, 0))
        attn_weights = self.softmax(attn_weights)

        # 加权平均
        attn_applied = torch.bmm(attn_weights.transpose(1, 2), enc_hidden)
        # 合并解码器输入和注意力上下文
        combined = torch.cat((dec_emb, attn_applied.expand(-1, dec_emb.size(1), -1)), dim=2)

        # 解码器RNN运行一步
        dec_output, state = self.decoder(combined, state)
        dec_output = self.mlp(dec_output)

        # 线性层
        sent_outputs = self.linear(dec_output)

        return sent_outputs, state
