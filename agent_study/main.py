# This is a sample Python script.
import ollama

def ask_local_code_model(prompt):
    response = ollama.chat(
        model="qwen3.5:2b",  # 换成你拉的模型名
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response["message"]["content"]

# 测试：让写Python排序代码
if __name__ == "__main__":
    res = ask_local_code_model("写一个Python快速排序算法，带详细注释")
    print(res)



