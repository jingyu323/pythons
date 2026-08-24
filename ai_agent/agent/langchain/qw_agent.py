from langchain.memory import ConversationBufferMemory
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor

# ========== 1. 定义智能体可用工具 ==========
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
# ========== 2. 初始化本地Qwen2.5-Coder模型 ==========
llm = ChatOllama(
    model="qwen2.5-coder:7b",
    temperature=0.1,  # 低温度，代码严谨、少幻觉
    num_ctx=32768    # 32K上下文，适配长代码
)

# 3. 记忆组件，memory_key 必须和 prompt 里 chat_history 对应
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True  # 返回 Message 对象，适配 MessagesPlaceholder
)
# 4. 提示词模板（保留chat_history，不再加optional）
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

# 5. 创建智能体
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    memory=memory  # 绑定记忆，自动填充chat_history
)
# ========== 5. 运行智能体测试 ==========
if __name__ == "__main__":
    # 多轮对话测试
    if __name__ == "__main__":
        # 第一轮对话
        res1 = agent_executor.invoke({"input": "计算 125 * 8"})
        print("回答1：", res1["output"])

        # 第二轮，会记住上一轮计算结果
        res2 = agent_executor.invoke({"input": "刚才的结果再加 1000 等于多少"})
        print("回答2：", res2["output"])

        # 查看完整历史
        print("\n===== 完整对话历史 =====")
        print(memory.load_memory_variables({})["chat_history"])
