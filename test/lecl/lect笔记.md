# lecl： 是一种在langchain之上封装的高级解释语言，简化链条开发支持真实生产环境而发明，极大的简化开发

##  LangSmith 可观测链路
LCEL 构建的链，打开追踪后，每一次请求会生成一条完整 Trace 树：

- 记录每一个 Runnable 节点：prompt、LLM 调用、工具调用、输出解析器
- 记录：输入、输出、耗时、token 消耗、错误栈
- 用途：调试 Agent / 复杂 RAG 链路、定位提示词问题、统计成本、评估效果、对比不同版本效果


1. 官方英文文档：https://python.langchain.com/docs/concepts/lcel/Langchain
2. 官方中文文档：https://python.langchain.ac.cn/docs/concepts/lcel/LangChain
3. LCEL 速查表（最常用）：https://python.langchain.ac.cn/docs/how_to/lcel_cheatsheet/LangChain
4. Agent 文档：https://python.langchain.com/docs/modules/agents/Langchain
## runtime 
- context  
- state agent 保存agent运行时产生的动态数据
- Store 存储夸会话的数据