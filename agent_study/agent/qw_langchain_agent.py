from langchain_ollama import ChatOllama

# 连接本地 qwen3.5:2b
llm = ChatOllama(
    model="qwen3.5:2b",
    base_url="http://localhost:11434",  # Ollama 默认地址
    temperature=0.7,
    num_ctx=4096  # 上下文窗口
)

# 测试调用
response = llm.invoke("你好，请自我介绍一下")
print(response.content)