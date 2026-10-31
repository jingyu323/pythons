from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# 1. 初始化嵌入模型
embeddings = OllamaEmbeddings(
    model="qwen3-embedding:0.6b",
    base_url="http://localhost:11434"
)

PERSIST_DIR = "./chroma_db"
# 2. 创建向量库
vectorstore = Chroma.from_texts(
    texts=[
        "LangChain 是大模型应用开发框架",
        "Chroma 是轻量级向量数据库",
        "Ollama 可以本地运行大模型",
        "RAG 通过检索增强大模型回答"
    ],
    embedding=embeddings,
    ids=["doc1", "doc2", "doc3", "doc4"],
    persist_directory=PERSIST_DIR
)

# 3. 查看文档数量
print("文档数量:", vectorstore._collection.count())

# 4. 基础相似度检索
docs = vectorstore.similarity_search("什么是向量数据库？", k=2)
for doc in docs:
    print("检索结果:", doc.page_content)

# 5. 带分数检索
results = vectorstore.similarity_search_with_relevance_scores(
    "什么是向量数据库？", k=2
)
for doc, score in results:
    print(f"分数: {score:.4f} | 内容: {doc.page_content}")

# 6. 按 ID 查询
print(vectorstore.get(ids=["doc1"]))

# 7. 增量添加
vectorstore.add_texts(
    texts=["向量数据库用于存储和检索文本向量"],
    ids=["doc5"]
)

# 8. 更新文档
# 正确写法
vectorstore.update_document(
    document_id="doc1",
    document=Document(
        page_content="LangChain 是 LLM 应用框架3333",
        metadata={"version": "updated"}
    )
)
print("文档数量5:", vectorstore._collection.count())

# 9. 删除文档
vectorstore.delete(ids=["doc5"])
print("文档数量-5:", vectorstore._collection.count())
print(vectorstore.get(ids=["doc1"]))
print(vectorstore)

# 11. 下次加载已有向量库
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)