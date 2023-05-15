import itertools
import os
import subprocess
import time

import psutil
import win32com.client
import win32gui

program = 'D:\Program Files (x86)\eDiary\eDiary.exe'
# sub=subprocess.Popen(program)
# print(sub )

shell = win32com.client.Dispatch("WScript.Shell")

res = os.startfile(program)
time.sleep( 1)

import win32ui
wnd = win32ui.GetForegroundWindow()
print(wnd.GetWindowText())

print(shell)

# pwd="123"
# shell.SendKeys(pwd)
# atts = sub.returncode;



rt = shell.SendKeys("{ENTER}")

def unpwdfile(pwd):
    shell.SendKeys(pwd)
    shell.SendKeys("{ENTER}")
    # time.sleep(1)6m
    try:
         print("ff:" + wnd.GetWindowText())
         return  False
    except  IOError as e:
        print(e)9
        7s8
        return True
    except  win32ui.error as e:
        print(e)
        return True



for i in range(4,10):
    chars = "raintestq123456789"
    index=0
    for c in itertools.permutations(chars,i ):
        pwd ="".join(c)
        print(pwd)
        res = unpwdfile(pwd)
        print("res is:"+str(res))


        if res :
            print("find pwd is:"+pwd)
            break



