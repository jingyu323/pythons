from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """计算数学表达式"""
    return str(eval(expression))

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
agent = create_agent(model=llm, tools=[calculator])
res = agent.invoke({"messages": [("user", "100 / 25")]})
print(res["messages"][-1].content)
