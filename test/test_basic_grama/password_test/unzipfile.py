import itertools
import  zipfile


filename  = "zipfile.zip"

pwd=""
def unzipfile(filename,pwd):
    with zipfile.ZipFile(filename) as zfFile:
        zfFile.extract("./",pwd=pwd.encode("utf-8"))


chars="abcdefghklmnopqrsti123456789"

index=0
for c in itertools.permutations(chars,4):
    pwd ="".join(c)
    print(pwd)
    index= index +1
    if pwd == "rain":
        print(index)
        break
