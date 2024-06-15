import torch
import tqdm
import numpy as np
import json
import os
from torch import nn
import torch.optim as optim

device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')


def read_json(file_path):
    ''' 读取 json 文件 '''
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def write_json(data, path):
    ''' 写入 json 文件 '''
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


raw_data = read_json('input/query_trainset.json')
data = [item for item in raw_data if item['evidence_list'] != []]


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

        self.fc = nn.Linear(hidden_size * 2, output_size)  # 注意这里乘以2，因为是双向的
        self.attention = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        x = x.unsqueeze(0)
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        # 计算注意力权重
        attention_w = torch.nn.functional.softmax(self.attention(out), dim=1)
        attention_out = torch.sum(attention_w * out, dim=1)

        out = self.fc(attention_out)
        return out


def retrieve_top_k_documents(query_embedding, document, model, k=3):
    with torch.no_grad():
        query_embedding = query_embedding.unsqueeze(0)
        document_embeddings = torch.stack([torch.tensor(item['facts_embedding'], device=device) for item in document])

        predict_embedding = model(query_embedding)
        similarities = torch.nn.functional.cosine_similarity(predict_embedding, document_embeddings, dim=-1)

        _, top_document_indices = similarities.topk(k)
        return top_document_indices.tolist()


def train_rnn_model(data, lr, epochs):
    # 初始化模型、损失函数和优化器
    input_size = 1024  # 输入维度
    hidden_size = 64
    output_size = 1024  # 输出维度
    model = LSTMModel(input_size, hidden_size, output_size)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练模型
    num_epochs = epochs
    for epoch in range(num_epochs):
        for item in data:
            query_embedding = item['query_embedding']
            query_embedding = torch.tensor(query_embedding, dtype=torch.float32)
            evidence_embeddings = [[doc['fact_embedding'] for doc in item['evidence_list']]]

            optimizer.zero_grad()
            query_embedding = query_embedding.unsqueeze(0)

            output = model(query_embedding)
            loss = 0
            for docs in evidence_embeddings:
                evidence_embedding = torch.tensor(docs, dtype=torch.float32)
                if not torch.all(evidence_embedding == 0):
                    loss += torch.norm(output - evidence_embedding, p=2)

            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

    return model


def zip_fun():
    path = os.getcwd()
    newpath = path + "/output/"
    os.chdir(newpath)
    os.system('zip prediction.zip result.json')
    os.chdir(path)


# 训练 RNN 模型
model = train_rnn_model(data, lr=0.001, epochs=25)

# 读取查询测试集和文档
test = read_json('input/query_testset.json')
test_embeddings = torch.tensor([entry['query_embedding'] for entry in test], device=device)

document = read_json('input/document.json')
document_embeddings = torch.tensor([entry['facts_embedding'] for entry in document], device=device)

results = []
for item in tqdm.tqdm(test):
    result = {}
    test_embedding = torch.tensor(item['query_embedding'], device=device)
    top_document_indices = retrieve_top_k_documents(test_embedding, document, model, k=3)
    result['query_input_list'] = item['query_input_list']
    result['evidence_list'] = [{'fact_input_list': document[index]['fact_input_list']} for index in
                               top_document_indices]
    results.append(result)

write_json(results, 'output/result.json')
print('write to output/result.json successful')
zip_fun()