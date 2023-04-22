import os
import subprocess

import win32com.client
program = 'D:\Program Files (x86)\eDiary\eDiary.exe'
sub=subprocess.Popen(program)
print(sub )
shell = win32com.client.Dispatch("WScript.Shell")

with open('dict_pass.txt', 'r') as f:
    while True:
        pwd = f.readline(end="")
        pwd=pwd.strip("\n")
        if not pwd:
            break
        if len(pwd)  > 3 :
            print(pwd)

            shell.SendKeys(pwd)
            print(shell)
            rt = shell.SendKeys("{ENTER}")
            print(rt)
            rt = shell.SendKeys("{ENTER}")




