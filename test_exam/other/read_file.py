def read_file_by_lines(file_path):
    """
    按行读取文件内容
    :param file_path: 文件路径
    :return: 行内容列表
    """
    try:
       with open(file_path, "r",encoding="utf-8") as file:
            lines = file.readlines()
            return lines
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 未找到")
        return []
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return []

def read_file_line_by_line(file_path):
    """
    逐行读取文件内容（适用于大文件）
    :param file_path: 文件路径
    :return: 生成器，逐行返回文件内容
    """
    try:
        with open(file_path, "r",encoding="utf-8") as file:
            for line in file:
                yield line.rstrip("\n")
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 未找到")
        return
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return


def display_file_content(file_path):
    """
    显示文件内容（带行号）
    :param file_path: 文件路径
    """
    lines = read_file_by_lines(file_path)
    if lines:
        print(f"文件 '{file_path}' 的内容:")
        print("-" * 50)
        for i, line in enumerate(lines, 1):
            print(f"{i:3d}: {line.rstrip()}")


if __name__ == "__main__":
    # 示例用法
    file_path = "ftp_log.txt"

    # 方法1: 一次性读取所有行
    print("\n方法1: 一次性读取所有行")
    display_file_content(file_path)

    # 方法2: 逐行读取（节省内存）
    print("\n方法2: 逐行读取（适用于大文件）")
    print("-" * 50)
    line_count = 0
    for line_num, line_content in enumerate(read_file_line_by_line(file_path), 1):
        print(f"{line_num:3d}: {line_content}")
        line_count = line_num

    if line_count > 0:
        print(f"\n文件共 {line_count} 行")