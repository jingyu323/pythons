from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# ---------------------- 1. 定义工具 ----------------------
@tool
def calc(expression: str) -> str:
    """数学计算器，支持加减乘除括号
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
    """读取本地代码文件
    Args:
        file_path: 文件路径
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"读取文件失败：{str(e)}"

tools = [calc, read_code_file]

# ---------------------- 2. 初始化本地Qwen模型 ----------------------
llm = ChatOllama(
    model="qwen2.5-coder:7b",
    temperature=0.1,
    num_ctx=32768
)
llm_with_tools = llm.bind_tools(tools)

# ---------------------- 3. 状态定义（存储所有对话消息） ----------------------
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# ---------------------- 4. 构建提示词 ----------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", """
你是专业代码智能体Qwen2.5-Coder。
规则：
1. 数学计算调用calc；读取本地代码调用read_code_file；
2. 必须使用工具获取真实数据，禁止编造；
3. 输出代码完整可运行，注释清晰。
"""),
    ("placeholder", "{messages}")
])

# ---------------------- 5. 节点函数 ----------------------
def agent_node(state: AgentState):
    chain = prompt | llm_with_tools
    resp = chain.invoke({"messages": state["messages"]})
    return {"messages": [resp]}

def tool_node(state: AgentState):
    from langgraph.prebuilt import ToolNode
    tool_exec = ToolNode(tools)
    return tool_exec.invoke(state)

# 判断是否继续调用工具
def should_continue(state: AgentState):
    last_msg = state["messages"][-1]
    if last_msg.tool_calls:
        return "tools"
    return END

# ---------------------- 6. 构建LangGraph流程图 + 内存持久化 ----------------------
# MemorySaver 内置对话记忆，替代废弃的 RunnableWithMessageHistory
memory = MemorySaver()
graph = StateGraph(AgentState)

graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

# 编译图并绑定记忆
graph_agent = graph.compile(checkpointer=memory)

# ---------------------- 7. 多轮对话测试 ----------------------
if __name__ == "__main__":
    # 会话ID，隔离不同用户对话历史
    config = {"configurable": {"thread_id": "chat_001"}}

    # 第一轮
    res1 = graph_agent.invoke(
        {"messages": [HumanMessage(content="计算 125 * 8")]},
        config=config
    )
    print("第一轮输出：\n", res1["messages"][-1].content, "\n")

    # 第二轮，自动读取历史上下文
    res2 = graph_agent.invoke(
        {"messages": [HumanMessage(content="刚才结果再加1000等于多少")]},
        config=config
    )
    print("第二轮输出：\n", res2["messages"][-1].content, "\n")

    # 查看完整对话历史
    all_msgs = graph_agent.get_state(config).values["messages"]
    print("====完整对话历史====")
    for m in all_msgs:
        print(f"{m.type}: {m.content}")

    # 清空当前会话历史
    # graph_agent.update_state(config, {"messages": []})