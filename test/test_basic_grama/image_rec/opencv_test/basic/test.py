

list = []


list.append("s")
list.append("s1")
list.append("s2")
list.append("s23")

for i in  list:
    print(i)

    if i =="s":
        list.remove(i)

        print(list)
        print(len(list))