from langchain.chat_models import init_chat_model

# 本地Ollama，model_provider填 ollama
llm = init_chat_model(
    model="qwen2.5-coder:7b",
    model_provider="ollama",
    temperature=0.2
)

print(type (llm))

res = llm.invoke("写一段Python快速排序代码")
print(res.content)