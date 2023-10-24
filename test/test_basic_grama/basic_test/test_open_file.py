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



print("======================================")

file1 = open('test.txt', 'r')
count = 0
while True:
  count = count + 1
  # Get next line from file
  s = file1.readline()

  # if line is empty
  # end of file is reached
  if not s:
    break
  print("Statement{}: {}".format(count, s.strip()))
file1.close()