import easyocr
# reader = easyocr.Reader(['ch_sim','en'], gpu=True,download_enabled=True) # this needs to run only once to load the model into memory
# reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)
reader = easyocr.Reader(['ch_sim', 'en'], gpu=True,download_enabled=False)
# result = reader.readtext('test.png')
result = reader.readtext('test2.jpg')

for res in result:
     print(res)
