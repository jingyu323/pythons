with open('test.txt', 'r') as f:
    print(f.read())

# 直接覆盖
with open('test.txt', 'a') as f:
    f.write('Hello, world111!\n')
    f.close()

# 追加
with open('test.txt', 'a') as f:
    f.write('Hello, world222!\n')
    f.close()



with open('test.txt', 'r') as f:
    for line in f:
        print(line)