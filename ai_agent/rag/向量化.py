from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="qwen3-embedding:0.6b",
    base_url="http://localhost:11434"
)

# 单条查询
vec = embeddings.embed_query("LangChain 是什么？")
print(vec)
print(f"向量维度: {len(vec)}")  # bge-m3 输出 1024 维
