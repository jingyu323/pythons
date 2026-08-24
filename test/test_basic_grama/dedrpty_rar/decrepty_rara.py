import os
import zipfile

import rarfile

def decryptRarZipFile(filename):
    if filename.endswith('.zip'):
        fp = zipfile.ZipFile(filename)
    elif filename.endswith('.rar'):
        fp = rarfile.RarFile(filename)
    desPath="."
    try:
        # 读取密码本文件
        fpPwd = open('pwd.txt')
    except:
        print('No dict file pwd.txt in current directory.')
        return
    index=0
    for pwd in fpPwd:
        pwd = pwd.rstrip()
        index = index +1
        try:
            fp.extractall(path=desPath, pwd=pwd.encode())
            print('Success! ====>' + pwd)
            fp.close()
            break
        except:
            pass
        print("pwd="+pwd+",index="+str(index))
    fpPwd.close()


if __name__ == '__main__':

    filename="project.rar"
    if os.path.isfile(filename) and filename.endswith(('.zip', '.rar')):
        decryptRarZipFile(filename)
    else:
        print('Must be Rar or Zip file')