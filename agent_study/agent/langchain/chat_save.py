import ollama
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver


def dp_oll_chat_save():
    # 创建智能体时指定checkpointer，LangChain会自动帮我们管理历史会话记忆
    agent = create_agent(
        "deepseek-chat",
        checkpointer=InMemorySaver()
    )

    # 设定thread_id，作为会话标识
    config = {"configurable": {"thread_id": "thread_1"}}

    # 第一次调用，告知AI我的信息
    response = agent.invoke(
        {"messages": [HumanMessage(content="你好，我叫虎哥，我最喜欢猫猫。")]},
        config  # 调用时添加thread_id，区分不同会话
    )

    print(response)


def ollama_stream_call(model: str, messages: list, options: dict, stream: bool = True):
    """封装函数：每次调用新建流，避免生成器耗尽问题"""
    return ollama.chat(
        model=model,
        messages=messages,
        stream=stream,
        options=options
    )

prompt_msg = [{"role": "user", "content": "介绍Transformer的GQA注意力"}]
opt = {"num_ctx": 32768, "temperature": 0.1}

# ========== 第一次流式调用 ==========
print("===== 第一次流式输出 =====")
stream1 = ollama_stream_call("qwen3.5:2b", prompt_msg, opt)
for chunk in stream1:
    content = chunk["message"]["content"]
    if content:
        print(content, end="", flush=True)
print("\n")

# ========== 第二次流式调用（重新发起请求，新建流） ==========
print("\n===== 第二次流式输出 =====")
stream2 = ollama_stream_call("qwen3.5:2b", prompt_msg, opt)
for chunk in stream2:
    content = chunk["message"]["content"]
    if content:
        print(content, end="", flush=True)
print("\n")

# ========== 一次性非流式调用（等价“invoke”效果） ==========
print("\n===== 非流式一次性完整返回 =====")
resp = ollama.chat(
    model="qwen3.5:2b",
    messages=prompt_msg,
    stream=False,
    options=opt
)
print(resp["message"]["content"])

