import torch
import tqdm
import json
import os
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 优先使用GPU


def read_json(file_path):
    '''读取 json 文件'''
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def write_json(data, path):
    '''写入 json 文件'''
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


raw_data = read_json('input/query_trainset.json')
data = [item for item in raw_data if item['evidence_list'] != []]


class CustomLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super(CustomLSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        x = x.unsqueeze(0)
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))

        attn_weights = torch.nn.functional.softmax(self.attn(out), dim=1)
        attn_output = torch.sum(attn_weights * out, dim=1)
        output = self.fc(attn_output)
        return output


def jaccard_similarity(query_embedding, document_embeddings):
    intersection = torch.sum(query_embedding * document_embeddings, dim=1)
    union = torch.sum(query_embedding + document_embeddings > 0, dim=1)
    jaccard = intersection / (union + 1e-8)  # 加上一个小值以避免除以零
    return jaccard


def retrieve_top_k_documents(query_embedding, document, model, k=3):
    with torch.no_grad():
        query_embedding = query_embedding.unsqueeze(0)
        document_embeddings = torch.stack([torch.tensor(item['facts_embedding'], device=device) for item in document])

        predict_embedding = model(query_embedding)
        cosine_similarities = F.cosine_similarity(predict_embedding, document_embeddings, dim=-1)
        jaccard_similarities = jaccard_similarity(query_embedding, document_embeddings)

        combined_scores = 0.5 * cosine_similarities + 0.5 * jaccard_similarities

        _, top_document_indices = combined_scores.topk(k)
        return top_document_indices.tolist()


input_size = 1024
hidden_size = 128
output_size = 1024
model = CustomLSTMModel(input_size, hidden_size, output_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

num_epochs = 1000
for epoch in range(num_epochs):
    total_loss = 0
    for item in data:
        query_embedding = torch.tensor(item['query_embedding'], dtype=torch.float32).to(device)
        evidence_embeddings = torch.stack(
            [torch.tensor(doc['fact_embedding'], dtype=torch.float32).to(device) for doc in item['evidence_list']])

        optimizer.zero_grad()
        query_embedding = query_embedding.unsqueeze(0)

        output = model(query_embedding)
        if evidence_embeddings.numel() > 0:
            loss = criterion(output, evidence_embeddings.mean(dim=0).unsqueeze(0))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    print(f'Epoch {epoch + 1}, Loss: {total_loss / len(data)}')

test = read_json('input/query_testset.json')
document = read_json('input/document.json')

results = []
for item in tqdm.tqdm(test):
    result = {}
    test_embedding = torch.tensor(item['query_embedding'], device=device)
    top_document_indices = retrieve_top_k_documents(test_embedding, document, model, k=3)
    result['query_input_list'] = item['query_input_list']
    result['evidence_list'] = [{'fact_input_list': document[index]['fact_input_list']} for index in
                               top_document_indices]
    results.append(result)

write_json(results, 'output/result200.json')
print('写入到 output/result.json 成功')


def zip_fun():
    path = os.getcwd()
    newpath = path + "/output/"
    os.chdir(newpath)
    os.system('zip prediction.zip result.json')
    os.chdir(path)


zip_fun()
