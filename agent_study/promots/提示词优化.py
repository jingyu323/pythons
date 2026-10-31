from langchain_classic.chains.constitutional_ai.prompts import examples
from openai import  OpenAI

# model = "qwen3.5:2b",
# base_url = "http://localhost:11434",

client = OpenAI(api_key='',
                base_url='http://localhost:11434/v1')

examples_data = {
    '新闻报道': '今日股市经历了一轮震荡，受到了宏观数据的影响',
}
examples_typs= ['新闻报道']

questions   =   ['今日，央行发布了降低利率以刺激经济增长']
meaages =[
    {"role":"system","content":"你是金融专家，将文本分类为['新闻报道','财务报道']，不清楚的就说不清楚类别"}
]

for question in questions:
    resp = client.chat.completions.create(
        model="qwen3.5:2b",
        messages=meaages,
    )
    print(resp.choices[0].message.content)