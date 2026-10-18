from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langchain.agents.output_parsers import ToolsAgentOutputParser
from langchain.agents.format_scratchpad import format_to_tool_messages
from langchain.agents import AgentExecutor

# 1. 定义工具
@tool
def calculator(expression: str) -> str:
    """计算数学表达式，例如：3*5+9，只接收数学表达式字符串"""
    return str(eval(expression))

@tool
def get_weather(city: str) -> str:
    """查询城市天气，入参是城市名"""
    return f"{city} 当前天气：晴，25℃"

tools = [calculator, get_weather]

# 2. 初始化模型，绑定工具
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# ========== 这里改动：bind_tools，不需要手动to_openai_function() ==========
llm_with_tools = llm.bind_tools(tools)

# 3. LCEL 组装Agent核心链路
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是智能助手，按需调用工具，得到结果后整理回答用户"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_tool_messages(x["intermediate_steps"])
    }
    | prompt
    | llm_with_tools
    | ToolsAgentOutputParser()
)

# 4. AgentExecutor 执行循环（思考/调用工具/观察）
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 5. 调用
resp = agent_executor.invoke({"input": "北京天气怎么样？12*24等于多少？"})
print(resp["output"])