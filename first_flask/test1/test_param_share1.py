import  test1
share_var = ""
def share_var1():
    global  share_var
    share_var = "32"
    print("share_var is :"+share_var)

    print(test1.test_param_share1)