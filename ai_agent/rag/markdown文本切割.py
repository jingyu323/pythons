from langchain_text_splitters import MarkdownHeaderTextSplitter

with open('rag_readme.md', 'r', encoding='utf-8') as f:
    doc = f.read()

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]


markdown_spliter  = MarkdownHeaderTextSplitter(headers_to_split_on,strip_headers=False)

chumks = markdown_spliter.split_text(doc)

for doc in chumks:
    print(doc.page_content)
    print(doc.metadata)
    print("=====================")
