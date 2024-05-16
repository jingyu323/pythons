import os

path='.'


def get_directory_size(directory):
    """
    获取目录的大小（以MB为单位）。
    """
    size = os.path.getsize(directory)
    return size / 1024 / 1024  # 转换为MB


for file in os.listdir(path):

    print(get_directory_size(file))
    if os.path.isdir(file):

        print(file.__sizeof__())
        os.path.getsize(file)
