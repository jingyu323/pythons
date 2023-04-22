import os
import subprocess
import psutil
import win32com.client
program = 'D:\Program Files (x86)\eDiary\eDiary.exe'
sub=subprocess.Popen(program)
print(sub )
shell = win32com.client.Dispatch("WScript.Shell")

with open('dict_pass.txt', 'r') as f:
    while True:
        pwd = f.readline().replace('\r','').replace('\n','')
        pwd=pwd.strip("\n")
        if not pwd:
            break
        if len(pwd)  > 3 :
            print(pwd)

            shell.SendKeys(pwd)
            atts = sub.returncode;

            print(atts)

            rt = shell.SendKeys("{ENTER}")
            print(rt)
            rt = shell.SendKeys("{ENTER}")

            # 杀死进程
            # for proc in psutil.process_iter():
            #     # res = re.findall("started='(\d.*)'", str(proc))
            #     if 'LicenseClient.exe' in proc.name():
            #         proc.kill()



