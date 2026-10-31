import numpy as np
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="qwen3-embedding:0.6b",
    base_url="http://localhost:11434"
)

# 单条查询
vec = embeddings.embed_query("LangChain 是什么？")
print(vec)
print(f"向量维度: {len(vec)}")  # bge-m3 输出 1024 维


texts = ["LangChain 是 LLM 应用框架", "Ollama 支持本地部署"]
vectorstore = Chroma.from_texts(
    texts=texts,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

results = vectorstore.similarity_search_with_score("LangChain 能做什么？", k=2)
for doc,score in results:
    print(doc,score)


# 你的余弦相似度函数
def cosine_similarity(vec1, vec2):
    """余弦相似度函数"""
    dot = np.dot(vec1, vec2)
    return dot / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# 测试
text1 = "今天天气真好"
text2 = "今天阳光明媚"
text3 = "我喜欢吃苹果"

vec1 = embeddings.embed_query(text1)
vec2 = embeddings.embed_query(text2)
vec3 = embeddings.embed_query(text3)

sim_1_2 = cosine_similarity(vec1, vec2)
sim_1_3 = cosine_similarity(vec1, vec3)

print(f"'{text1}' 和 '{text2}' 的相似度: {sim_1_2:.4f}") # 结果应该接近 1
print(f"'{text1}' 和 '{text3}' 的相似度: {sim_1_3:.4f}") # 结果应该接近 0



# =====================
# 2. 准备测试文本
# =====================
text1 = "今天天气真好"
text2 = "今天阳光明媚"
text3 = "我喜欢吃苹果"

# =====================
# 3. 把 text2、text3 放入向量库
# =====================
vectorstore = Chroma.from_texts(
    texts=[text2, text3],
    embedding=embeddings
)

# =====================
# 4. 用 similarity_search 替代原来的 cosine_similarity
# =====================
results = vectorstore.similarity_search_with_relevance_scores(
    text1,
    k=2
)

print(f"查询文本: '{text1}'\n")

for doc, score in results:
    print(f"'{text1}' 和 '{doc.page_content}' 的相似度: {score:.4f}")

print("\n说明：")
print("- 这里不再手动算 cosine_similarity")
print("- 而是把 text2、text3 放进向量库")
print("- 再用 similarity_search_with_relevance_scores 检索 text1")
print("- 返回的 score 一般是 0~1，越大越相似")