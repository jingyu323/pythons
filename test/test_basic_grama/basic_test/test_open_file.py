with open('test.txt', 'r') as f:
    print(f.read())

# 直接覆盖
with open('test.txt', 'w') as f:
    f.write('Hello, world!')

# 追加
with open('test.txt', 'a') as f:
    f.write('Hello, world!')

    while True:
        str = f.readline()

        if not str:
            break
        print(str)

        if str.__contains__("s"):
            print("包含")
