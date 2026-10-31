"""
RAG 在Agent 中的作用
用户提问
→
→ 知识检索（问题向量化 + 检索知识片段）
→
→ 增强提示词（把检索到的资料拼接到你的问题里）
→
→ 调用模型（让模型根据增强后的提示词生成回答）

合适使用Rag 增强
-  每次调用
- 模型自主决定  绑定很多工具研究助手
- 结合两种模式
"""
from operator import itemgetter
from typing import TypedDict, Literal

from langchain_chroma import Chroma
from langchain_classic.agents import create_openai_functions_agent, AgentExecutor
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import Tool, create_retriever_tool
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.agents import create_agent
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.prebuilt import create_react_agent


# 1. 定义路由类型
class RouteQuery(TypedDict):
    destination: Literal["policy_chain", "general_chain"]


# 1. 准备嵌入模型 (假设你本地 Ollama 已运行 qwen3-embedding:0.6b)
embeddings = OllamaEmbeddings(
    model="qwen3-embedding:0.6b",
    base_url="http://localhost:11434"
)

# 2. 定义本地持久化目录
PERSIST_DIR = "./chroma_db"

# 3. 初始化并创建向量库
# from_texts 方法会自动计算文本向量并存入数据库
vectorstore = Chroma.from_texts(
    texts=[
        "LangChain 是大模型应用开发框架",
        "Chroma 是轻量级向量数据库",
        "Ollama 可以本地运行大模型",

        "RAG 通过检索增强大模型回答",
        "公司的年假规定12天",
        "生病需要三甲医院的诊断证明",
        "餐饮每天额度是30元",
    ],
    embedding=embeddings,
    ids=["doc1", "doc2", "doc3", "doc4", "doc5", "doc6", "doc7"],
    persist_directory=PERSIST_DIR  # 核心参数：指定本地保存路径
)

# 验证：打印文档数量
print(f"新建库成功，当前文档数: {vectorstore._collection.count()}")
retriever = vectorstore.as_retriever()
# --- 医院规定库 ---
hospital_docs = [
    Document(page_content="门诊挂号时间为每日8:00-17:00，急诊24小时开放。"),
    Document(page_content="住院押金标准：普通病房2000元，ICU 5000元。"),
    Document(page_content="探视时间：每日14:00-16:00，ICU需提前预约。"),
    Document(page_content="住院患者请假需主治医生批准并提交书面申请，单次不得超过24小时。"),
    Document(page_content="开具病假证明需携带身份证、病历本，由接诊医生根据病情判断是否出具。"),
]
hospital_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
hospital_chunks = hospital_splitter.split_documents(hospital_docs)

hospital_vectorstore = Chroma(
    collection_name="hospital_rules",
    embedding_function=embeddings,
    persist_directory="./chroma_hospital"
)
hospital_vectorstore.add_documents(hospital_chunks)
hospital_retriever = hospital_vectorstore.as_retriever(search_kwargs={"k": 2})

# ===================== 3. 合并检索器 =====================
combined_retriever = EnsembleRetriever(
    retrievers=[retriever, hospital_retriever],
    weights=[0.5, 0.5]
)

llm = ChatOllama(
    model="qwen3:4b",  # 改成你本地 ollama list 里实际存在的模型名
    base_url="http://localhost:11434",
    temperature=0.2,  # 越低越稳定，适合问答、代码、RAG
    num_predict=1024,  # 最大生成 token 数
    timeout=60,  # 本地模型首次加载可能较慢
)


def initialize_agent(tools, llm):
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt="你是公司政策助手。涉及公司制度问题时，必须先调用 search_company_policy 查询，再根据查询结果回答。不要编造。",
    )


"""
在 AI Agent（智能体）架构中，调用知识库主要有以下 3 种主流方式。
这三种方式在实现逻辑、响应速度和准确度上各有优劣：
方式 1：直接嵌入 Agent 工作流（原生检索）
核心逻辑：在 Agent 的代码逻辑中，强制或直接调用向量检索。无论用户问什么，Agent 都必须先去数据库查一遍。
优点：实现最简单，绝对保证会去查资料。
缺点：不够灵活，用户问“今天天气怎么样”这种通用问题，它也会去业务库里瞎找，浪费时间和算力。
LangChain 示例代码：
直接将检索器（Retriever）包装成 Agent 的工具。

"""


def embding():
    # 假设已经初始化了 vectorstore

    retriever_tool = create_retriever_tool(
        retriever,
        "search_company_policy",
        "查询公司政策、规章制度、年假、报销、流程等信息。",
    )
    hospital_retriever_tool = create_retriever_tool(
        hospital_retriever,
        "search_hospital_policy",
        "查询医院政策、规章制度、住院、报销、门诊、流程等信息。",
    )

    # 将知识库检索绑定为一个 Agent 工具
    tools = [
        Tool(
            name="知识库检索",
            func=lambda query: retriever.invoke(query),
            description="查询公司内部政策、规章制度、年假、报销、流程等信息。输入应为自然语言问题时使用此工具。"
        ),
        hospital_retriever_tool,
        retriever_tool,
    ]

    docs = retriever.invoke("公司年假规定")
    # for d in docs:
    #     print(d.page_content)

    # 初始化 Agent
    # 运行 Agent
    agent = initialize_agent(tools, llm)
    result = agent.invoke({
        "messages": [("user", "公司的年假规定是多少天？")]
    })
    print("======")
    print(result["messages"][-1].content)
    print("======222")
    # 打印所有消息
    for i, msg in enumerate(result["messages"]):
        print(f"--- 第 {i + 1} 条消息 ---")
        print(f"  类型: {msg.type}")
        if hasattr(msg, "name") and msg.name:
            print(f"  工具名: {msg.name}")
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            print(f"  工具调用: {msg.tool_calls}")
        print(f"  内容: {msg.content}")
        print()


"""

核心逻辑：在检索前增加一个“分类器”或“路由层”。Agent 先分析用户的意图，如果是业务相关问题，再自动决定调用知识库；如果是闲聊，则直接走大模型。
优点：非常灵活，能精准区分“闲聊”、“代码解释”、“知识检索”等不同场景，避免无效检索。
缺点：对 Prompt 或路由模型的分类准确率有一定要求。
LangChain 示例代码：
使用 Router 或 MultiPromptChain 进行意图分发。
"""


def autoRouter():
    # 2. 定义子链
    policy_prompt = ChatPromptTemplate.from_template(
        "你是一个公司政策专家。结合上下文回答问题。\n上下文:\n{context}\n问题:{input}"
    )
    policy_chain = policy_prompt | llm | StrOutputParser()

    general_prompt = ChatPromptTemplate.from_template(
        "你是一个友好的助手。请用自然的语气回答。\n问题:{input}"
    )
    general_chain = general_prompt | llm | StrOutputParser()

    # 3. 路由链：让模型判断走哪条分支
    route_prompt = ChatPromptTemplate.from_template(
        "根据以下描述，将问题路由到最合适的目的地。\n"
        "- policy_chain: 关于公司政策、规章制度、福利、请假等问题。\n"
        "- general_chain: 聊天、翻译、写代码等通用问题。\n"
        "问题: {input}"
    )

    route_chain = (
            route_prompt

            | llm.with_structured_output(RouteQuery)
            | itemgetter("destination")
    )

    # 4. 分支选择（关键修复）
    # def select_chain(inputs: dict):
    #     destination = inputs["destination"]
    #     user_input = inputs["input"]
    #
    #     if destination == "policy_chain":
    #         docs = retriever.invoke(user_input)
    #         context = "\n".join(d.page_content for d in docs)
    #         return policy_chain.invoke({"input": user_input, "context": context})
    #     else:
    #         return general_chain.invoke({"input": user_input})
    #
    # chain = {
    #             "destination": route_chain,
    #             "input": lambda x: x["input"],
    #         } | RunnableLambda(select_chain)

    # 5. 调用
    # result = chain.invoke({"input": "我生病请假需要提交什么证明？"})
    # print(result)

    # 4. 流式分支选择
    def stream_select_chain(inputs: dict):
        destination = inputs["destination"]
        user_input = inputs["input"]

        if destination == "policy_chain":
            docs = retriever.invoke(user_input)
            context = "\n".join(d.page_content for d in docs)
            for chunk in policy_chain.stream({"input": user_input, "context": context}):
                yield chunk
        else:
            for chunk in general_chain.stream({"input": user_input}):
                yield chunk

    chain = {
                "destination": route_chain,
                "input": lambda x: x["input"],
            } | RunnableLambda(stream_select_chain)

    # 5. 流式调用
    for chunk in chain.stream({"input": "我生病请假需要提交什么证明？"}):
        print(chunk, end="", flush=True)


def rag_funcation_cal():
    # 1. 将检索器包装成 Tool，并给一个明确的描述
    retriever_tool = create_retriever_tool(
        retriever,
        "search_company_policy",
        "当你需要查询具体的企业政策、规章制度或内部文档时使用此工具。输入应为详细的搜索查询词。"
    )
    hos_retriever_tool = create_retriever_tool(
        hospital_retriever,
        "search_hospital_policy",
        "查询医院政策、规章制度、住院、报销、门诊、流程等信息"
    )

    tools = [retriever_tool, hos_retriever_tool]

    system_prompt = (
        "你是公司政策助手，负责解答员工关于规章制度、报销、请假、福利等方面的问题。\n"
        "你可以使用 search_company_policy 工具查询企业内部文档。\n"
        "规则：\n"
        "1. 涉及具体政策、金额、流程的问题，必须先调用工具查询，不能凭空编造。\n"
        "2. 回答要简洁准确，引用政策原文时注明出处。\n"
        "3. 如果工具返回的内容不足以回答问题，请如实告知用户未找到相关信息。"
    )

    # 2. 使用 OpenAI 函数调用能力创建 Agent

    # 3. Agent 运行
    # 模型会自主判断：这个问题需要查资料 -> 触发 search_policy_database 函数 -> 得到结果 -> 总结回答

    # result = agent.invoke({
    #     "messages": [HumanMessage(content="报销流程中，餐饮发票的报销额度是多少？")]
    # })
    # print("======")
    # print(result["messages"][-1].content)
    # print("======222")

    # 用 LangGraph 创建 agent（替代 create_openai_functions_agent + AgentExecutor）
    # 新版入口，必须用关键字参数
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
    )

    # 流式调用
    # for event_type, chunk in agent.stream(
    #         {"messages": [HumanMessage(content="报销流程中，餐饮发票的报销额度是多少？")]},
    #         stream_mode="messages",
    # ):
    #     if chunk:
    #         print(chunk, end="", flush=True)

    # result = agent.invoke({
    #     "messages": [HumanMessage(content="报销流程中，餐饮发票的报销额度是多少？")]
    # })
    # for msg in result["messages"]:
    #     print(msg.type, ":", msg.content)

    for chunk in agent.stream(
            {"messages": [HumanMessage(content="报销流程中，餐饮发票的报销额度是多少？"),
                          HumanMessage(content="住院押金标准是多少？")]},
            stream_mode="messages",
            version="v2",
    ):
        if chunk.get("type") != "messages":
            continue
        token, metadata = chunk["data"]
        if token.content:
            print(token.content, end="", flush=True)


if __name__ == '__main__':
    rag_funcation_cal()
