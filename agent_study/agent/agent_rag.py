import ollama

# --------------------------
# 可配置参数（根据你的模型调整）
# --------------------------
MODEL_NAME = "qwen3.5:2b"  # 换成 qwen3.5-coder:4b（显存≥8GB）
MAX_RETRY = 2  # 自我纠错最大次数（避免死循环）
CODE_PROMPT_TEMPLATE = """
你是一个专业的Python代码Agent，负责：
1. 先拆解用户需求（1-2句话说清步骤）；
2. 生成完整、可运行的Python代码，带详细注释；
3. 生成后自我检查：代码是否可运行、是否符合需求、有无语法错误。

用户需求：{user_prompt}
请先输出拆解步骤，再输出代码，最后输出检查结果。
"""

# 自我纠错提示模板
CORRECT_PROMPT_TEMPLATE = """
你生成的代码存在问题（{error_info}），请进行以下操作：
1. 指出问题所在（简洁明了）；
2. 修正代码，保证可运行、符合原需求；
3. 再次检查修正后的代码，确认无错误。
原需求：{user_prompt}
原代码：{original_code}
"""


class LocalCodeAgent:
    def __init__(self, model_name=MODEL_NAME):
        self.model = model_name
        self.retry_count = 0  # 纠错次数计数

    def _call_model(self, prompt):
        """调用本地Qwen模型，获取响应"""
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response["message"]["content"]
        except Exception as e:
            return f"模型调用失败：{str(e)}"

    def _parse_response(self, response):
        """解析模型响应，提取 拆解步骤、代码、检查结果"""
        # 简单解析（适配Qwen输出格式，无需复杂正则）
        steps = ""
        code = ""
        check_result = ""

        if "拆解步骤" in response:
            steps = response.split("拆解步骤：")[-1].split("代码：")[0].strip()
        if "代码：" in response:
            code_block = response.split("代码：")[-1].split("检查结果：")[0].strip()
            # 提取 ```python 包裹的代码（避免多余文本）
            if "```python" in code_block:
                code = code_block.split("```python")[1].split("```")[0].strip()
            else:
                code = code_block
        if "检查结果" in response:
            check_result = response.split("检查结果：")[-1].strip()

        return steps, code, check_result

    def _self_correct(self, user_prompt, original_code, error_info):
        """自我纠错：根据错误信息修正代码"""
        if self.retry_count >= MAX_RETRY:
            return f"已达到最大纠错次数（{MAX_RETRY}次），无法修正，原错误：{error_info}", None

        self.retry_count += 1
        correct_prompt = CORRECT_PROMPT_TEMPLATE.format(
            error_info=error_info,
            user_prompt=user_prompt,
            original_code=original_code
        )
        correct_response = self._call_model(correct_prompt)
        _, corrected_code, _ = self._parse_response(correct_response)
        return corrected_code, self.retry_count

    def run(self, user_prompt):
        """核心运行流程：拆解→生成→检查→纠错"""
        print(f"【代码Agent开始工作】需求：{user_prompt}\n")

        # 1. 生成代码（含拆解、检查）
        prompt = CODE_PROMPT_TEMPLATE.format(user_prompt=user_prompt)
        response = self._call_model(prompt)
        steps, code, check_result = self._parse_response(response)

        # 2. 输出拆解步骤
        print(f"【任务拆解】\n{steps}\n")

        # 3. 检查代码是否可用，若有错误则纠错
        if not code:
            print("❌ 生成代码失败，无有效代码输出")
            return

        print(f"【初始生成代码】\n{code}\n")
        print(f"【自我检查结果】\n{check_result}\n")

        # 简单模拟代码运行检查（实际可执行代码校验，这里简化）
        # 可根据需求替换为：exec(code) 实际运行校验（注意安全）
        if "错误" in check_result or "无法运行" in check_result:
            print(f"❌ 代码存在问题，开始第{self.retry_count + 1}次纠错...\n")
            corrected_code, retry_num = self._self_correct(user_prompt, code, check_result)
            if corrected_code:
                print(f"【纠错后代码】（第{retry_num}次）\n{corrected_code}\n")
                print("✅ 纠错完成，建议手动验证代码运行效果")
            else:
                print("❌ 纠错失败，请调整需求或更换更高参模型（如4B）")
        else:
            print("✅ 代码检查通过，可直接运行")

        return code


# --------------------------
# 测试运行（替换成你的需求）
# --------------------------
if __name__ == "__main__":
    # 实例化Agent
    code_agent = LocalCodeAgent(model_name=MODEL_NAME)

    # 测试需求（可替换成任意Python代码需求）
    test_prompt = "写一个Python脚本，批量读取指定文件夹下所有Excel文件，合并成一个新的Excel，保存到当前目录，带进度条显示"

    # 运行Agent
    code_agent.run(test_prompt)
