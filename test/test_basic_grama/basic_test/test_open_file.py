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



print("==========单行读取==========readline==================")

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


print("==========多行读取======readlines======================")


file = open('test.txt', 'r')
Statements = file.readlines()
count = 0
# Strips the newline character
for line in Statements:                                     # Using for loop to print the data of the file
  count = count + 1
  print("Statement{}: {}".format(count, line.strip()))