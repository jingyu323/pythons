

# 最长子串
def max_sub_str():
    str = "abcbcefdds"
    start = 0
    end=0
    max_len=0
    for i in range(len(str)):

        if str[i] in str[start:end]:

            l = len(str[start:end])
            if  l > max_len:
                max_len = l
                start=end

        else:
            l = len(str[start:end])
            if l > max_len:
                max_len = l
            end = i+1

    print(max_len)

def  max_sub_str2():
    str = "abcbcefdds"
    start = 0
    end = 0
    max_len = 0

    for i in range(len(str)):
        if str[i] in str[start:end]:
            if i  -start >max_len:
                max_len = i-start
                start = end
        else:
            end = i+1
    # if len(str) - end > max_len:
    #     max_len = len(str) - start

    return max_len





if __name__ == '__main__':
    max_sub_str()
    l=  max_sub_str2()
    print(l)
