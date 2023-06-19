
from bs4 import  BeautifulSoup

soup = BeautifulSoup('<p>Hello World</p>',"lxml")

print(soup.p.string)

# import tesserocr
import pytesseract
from PIL import Image
path="E:\\git_project\\pythons\\test\\test_basic_grama\\test_spider\\test1.png"
image=Image.open(path)
text=pytesseract.image_to_string(image,lang='chi_sim')
print(text)

