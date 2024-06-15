import torch
import tqdm
import numpy as np
import json
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def read_json(file_path):
    ''' 读取 json 文件 '''
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def write_json(data, path):
    ''' 写入 json 文件 '''
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

def retrieve_top_k_documents(query_embedding, document_embeddings, k=3):
    """
    从所有document embeddings中检索出与query embedding最相关的前k个document。
    Args:
        query_embedding: Query的embedding向量，大小为(N,)，N为embedding的维度。
        document_embeddings: Document的embedding向量列表，每个向量的大小为(N,)，N为embedding的维度。
        k: 要检索的top k个document。
    Returns:
        top_documents: 一个列表，包含与query最相关的前k个document的索引。
    """
    similarities = torch.nn.functional.cosine_similarity(query_embedding.unsqueeze(0), document_embeddings, dim=-1)
    # 使用topk获取排序后的索引，然后选择前k个最大的相似度值对应的document索引
    _, top_document_indices = similarities.topk(k)
    return top_document_indices.tolist()

def zip_fun():
    path=os.getcwd()
    newpath=path+"/output/"
    os.chdir(newpath)
    os.system('zip prediction.zip result.json')
    os.chdir(path)

# 读取query_testset文件（512条）
query = read_json('input/query_testset.json')
query_embeddings = torch.tensor([entry['query_embedding'] for entry in query], device=device)

# 读取检索fact（26599条）
document = read_json('input/document.json')
document_embeddings = torch.tensor([entry['facts_embedding'] for entry in document], device=device)

results = []
for item in tqdm.tqdm(query):
    result = {}
    query_embedding = torch.tensor(item['query_embedding'], device=device)
    top_document_indices = retrieve_top_k_documents(query_embedding, document_embeddings, k=3)
    result['query_input_list'] = item['query_input_list']
    result['evidence_list'] = [{'fact_input_list': document[index]['fact_input_list']} for index in top_document_indices]
    results.append(result)

write_json(results, 'output/result.json')
print('write to output/result.json successful')
zip_fun()
