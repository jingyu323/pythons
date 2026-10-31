#!/usr/bin/python
# -*- coding: UTF-8 -*-

import re


def extract_subdirectories(file_content):
    """
    从FTP日志内容中提取6A/后的第一个子目录名称
    """
    # 正则表达式匹配模式：6A/后面跟着非斜杠字符（一个或多个），直到遇到下一个斜杠
    pattern = r'6A/([^/]+)/'

    # 使用findall查找所有匹配项
    matches = re.findall(pattern, file_content)

    # 去重并返回结果
    unique_matches = list(set(matches))
    return unique_matches


def read_and_extract(file_path):
    """
    读取文件并提取子目录信息
    """
    try:
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # 提取子目录
        subdirectories = extract_subdirectories(content)

        # 输出结果
        print(f"找到 {len(subdirectories)} 个不同的子目录：")
        print("-" * 50)

        # 按字母顺序排序输出
        for subdir in sorted(subdirectories):
            print(f"  • {subdir}")

        return subdirectories

    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 未找到")
        return []
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return []


# 主程序
if __name__ == "__main__":
    # 如果您要读取本地文件，请修改这里的文件路径
    file_path = "ftp_log.txt"  # 或者使用完整的文件路径

    # 如果文件在当前目录，直接运行
    # 否则，可以取消下面的注释，让用户输入文件路径
    # file_path = input("请输入FTP日志文件路径: ")

    subdirectories = read_and_extract(file_path)

    print(subdirectories)
