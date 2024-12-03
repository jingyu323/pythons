import os

from PyPDF2 import PdfFileMerger, PdfMerger

# 创建一个 PdfFileMerger 对象
merger = PdfMerger()

path_dir  = "d://Downloads"

paths = os.listdir(path_dir)

pdf_files = [ ]
for path in paths:
    if path.endswith(".pdf")  and  path.find("12083号")>0:


        file_path = path_dir+"/"+path
        pdf_files.append(file_path)
        print(file_path)


# # 添加要合并的 PDF 文件
for pdf_file in pdf_files:
    merger.append(pdf_file)

# 指定输出文件路径
output_path = 'merged.pdf'

# 执行合并操作
merger.write(output_path)

# 关闭 PdfFileMerger 对象
merger.close()