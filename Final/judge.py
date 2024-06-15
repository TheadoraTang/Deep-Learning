from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

df = pd.read_csv('dataset.csv', encoding='latin1')
df = df.head(6)
# 定义一个函数来计算余弦相似度
def cosine_similarity_score(text1, text2):
    tfidf_vectorizer = TfidfVectorizer().fit_transform([text1, text2])
    return cosine_similarity(tfidf_vectorizer[0:1], tfidf_vectorizer[1:2])[0][0]

# 计算每一列与Correct列的余弦相似度评分
# df['Cosine_Similarity_Corrected_languagetool'] = df.apply(lambda row: cosine_similarity_score(row['Correct'], row['Corrected_languagetool']), axis=1)
# df['Cosine_Similarity_Corrected_tencent'] = df.apply(lambda row: cosine_similarity_score(row['Correct'], row['Corrected_tencent']), axis=1)
# df['Cosine_Similarity_Corrected_trinka'] = df.apply(lambda row: cosine_similarity_score(row['Correct'], row['Corrected_trinka']), axis=1)
df['Cosine_Similarity_Corrected_NLPCloud'] = df.apply(lambda row: cosine_similarity_score(row['Correct'], row['Corrected_NLPCloud']), axis=1)

# # 打印相似度评分
# print(df[['Correct', 'Corrected_languagetool', 'Cosine_Similarity_Corrected_languagetool', 'Corrected_tencent', 'Cosine_Similarity_Corrected_tencent', 'Corrected_trinka', 'Cosine_Similarity_Corrected_trinka']])

df.to_csv('judge——2.0.csv', index=False)