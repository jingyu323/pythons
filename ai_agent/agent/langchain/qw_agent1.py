from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# 1. 工具定义
@tool
def calc(expression: str) -> str:
    """数学计算器，支持加减乘除括号，输入纯数学表达式
    Args:
        expression: 数学算式，例如 "123 * 45 + 78"
    """
    try:
        res = eval(expression)
        return f"计算结果：{res}"
    except Exception as e:
        return f"计算失败，错误：{str(e)}"

@tool
def read_code_file(file_path: str) -> str:
    """读取本地代码文件内容，用于分析、修改代码
    Args:
        file_path: 文件绝对/相对路径
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"读取文件失败：{str(e)}"

tools = [calc, read_code_file]

# 2. 本地Qwen2.5-Coder模型
llm = ChatOllama(
    model="qwen2.5-coder:7b",
    temperature=0.1,
    num_ctx=32768
)

# 3. Prompt模板（保留chat_history占位）
prompt = ChatPromptTemplate.from_messages([
    ("system", """
你是专业代码助手智能体，基于Qwen2.5-Coder。
规则：
1. 需要计算数字时调用calc工具；
2. 需要查看本地代码文件时调用read_code_file；
3. 工具返回结果后再整合答案，不要编造数据；
4. 输出代码规范、可直接运行，注释清晰。
"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# 4. 创建基础Agent
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 5. 新版对话历史管理（替代废弃的ConversationBufferMemory）
# 存储会话ID对应的消息历史
store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 绑定历史链路
agent_with_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

# 多轮对话测试
if __name__ == "__main__":
    session_cfg = {"configurable": {"session_id": "session_001"}}

    # 第一轮
    res1 = agent_with_history.invoke({"input": "计算 125 * 8"}, config=session_cfg)
    print("回答1：", res1["output"])

    # 第二轮，自动读取历史上下文
    res2 = agent_with_history.invoke({"input": "刚才的结果再加1000等于多少"}, config=session_cfg)
    print("回答2：", res2["output"])

    # 查看当前会话全部历史
    hist = get_session_history("session_001")
    print("\n完整对话历史：")
    for msg in hist.messages:
        print(f"{msg.type}: {msg.content}")