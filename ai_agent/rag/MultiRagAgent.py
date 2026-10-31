import os
from operator import itemgetter
from typing import Literal

from langchain_classic.retrievers import EnsembleRetriever
from typing_extensions import TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings

# ===================== 1. 模型与嵌入 =====================
llm = ChatOllama(model="qwen3:8b", temperature=0)
embeddings = OllamaEmbeddings(
    model="qwen3-embedding:0.6b",
    base_url="http://localhost:11434"
)

# ===================== 2. 构建多个知识库 =====================

# --- 公司制度库 ---
company_docs = [
    Document(
        page_content="公司员工病假需提交二级以上医院开具的诊断证明及病假建议书，3天以内由部门经理审批，3天以上需人力资源部备案。"),
    Document(page_content="事假需提前3个工作日提交书面申请，经直属上级批准后方可休假。"),
    Document(page_content="年假按工龄计算：1-5年5天，5-10年10天，10年以上15天。"),
    Document(page_content="迟到30分钟以内不计考勤，超过30分钟按半天事假处理。"),
]
company_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
company_chunks = company_splitter.split_documents(company_docs)

company_vectorstore = Chroma(
    collection_name="company_rules",
    embedding_function=embeddings,
    persist_directory="./chroma_company"
)
company_vectorstore.add_documents(company_chunks)

company_retriever = company_vectorstore.as_retriever(search_kwargs={"k": 2})

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
    retrievers=[company_retriever, hospital_retriever],
    weights=[0.5, 0.5]
)

# ===================== 4. 子链定义 =====================
policy_prompt = ChatPromptTemplate.from_template(
    "你是一个公司政策专家。结合上下文回答问题。\n上下文:\n{context}\n问题:{input}"
)
policy_chain = policy_prompt | llm | StrOutputParser()

general_prompt = ChatPromptTemplate.from_template(
    "你是一个友好的助手。请用自然的语气回答。\n问题:{input}"
)
general_chain = general_prompt | llm | StrOutputParser()


# ===================== 5. 路由链 =====================
class RouteQuery(TypedDict):
    destination: Literal["policy_chain", "general_chain"]


route_prompt = ChatPromptTemplate.from_template(
    "根据以下描述，将问题路由到最合适的目的地。\n"
    "- policy_chain: 关于公司政策、规章制度、福利、请假、医院规定等问题。\n"
    "- general_chain: 聊天、翻译、写代码等通用问题。\n"
    "问题: {input}"
)

route_chain = (
        route_prompt

        | llm.with_structured_output(RouteQuery)
        | itemgetter("destination")
)


# ===================== 6. 流式分支选择 =====================
def stream_select_chain(inputs: dict):
    destination = inputs["destination"]
    user_input = inputs["input"]

    print(f"\n[路由结果] {destination}")

    if destination == "policy_chain":
        docs = combined_retriever.invoke(user_input)
        print(f"[检索到 {len(docs)} 条文档]")
        context = "\n".join(d.page_content for d in docs)
        if not context:
            context = "未检索到相关规定，请基于通用知识回答。"
        for chunk in policy_chain.stream({"input": user_input, "context": context}):
            print(chunk, end="", flush=True)
    else:
        for chunk in general_chain.stream({"input": user_input}):
            print(chunk, end="", flush=True)


chain = {
            "destination": route_chain,
            "input": lambda x: x["input"],
        } | RunnableLambda(stream_select_chain)

# ===================== 7. 调用示例 =====================
if __name__ == "__main__":
    questions = [
        "我生病请假需要提交什么证明？",
        "住院患者能请假外出吗？",
        "帮我写一首诗",
    ]
    for q in questions:
        print(f"\n{'=' * 50}")
        print(f"问题: {q}")
        print("=" * 50)
        for _ in chain.stream({"input": q}):
            pass
        print()
