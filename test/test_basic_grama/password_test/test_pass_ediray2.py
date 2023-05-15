import itertools
import os
import subprocess
import psutil
import win32com.client
program = 'D:\Program Files (x86)\eDiary\eDiary.exe'
sub=subprocess.Popen(program )
print(sub )
shell = win32com.client.Dispatch("WScript.Shell")


def unpwdfile(pwd):
    shell.SendKeys(pwd)
    atts = sub.returncode;
    print(atts)
    rt = shell.SendKeys("{ENTER}")

    print(rt)

    rt = shell.SendKeys("{ENTER}")
    print(rt)



for i in range(3,10):


    chars = "abcdefghklmnopqrsti123456789"
    index=0
    for c in itertools.permutations(chars,i ):
        pwd ="".join(c)
        print(pwd)
        unpwdfile(pwd)





pwd=""

# unpwdfile(pwd)



