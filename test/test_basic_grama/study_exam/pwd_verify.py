def jiaoyan(x):
    shu =[0,"1","2","3","4","5","6","7","8","9"]
    xiao ="abcdefghijklmnopqrstuvwxyz"
    da ="ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    xiao =list(xiao)
    da =list(da)
    if len(x) <=8 or " " in x:
        print("NG")
    else:
        x =list(x)
        b =[]
        for i in x:
            if i in xiao:
                b.append("xiao")
            elif i in da:
                b.append("da")
            elif i in shu:
                b.append("shu")
            else:
                b.append("fuhao")
        b =set(b)
        if len(b) <3:
            print("NG")
        else:
            print("OK")
a =input()
jieguo =jiaoyan(a)
print(jieguo)