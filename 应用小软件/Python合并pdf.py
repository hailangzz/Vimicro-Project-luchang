import os
from PyPDF2 import PdfFileMerger


target_path = r'C:\Users\34426\Documents\树萍文档'  ## pdf目录文件
pdf_lst = [f for f in os.listdir(target_path) if f.endswith('.pdf')]
pdf_lst = [os.path.join(target_path, filename) for filename in pdf_lst]


file_merger = PdfFileMerger()
for pdf in pdf_lst:
    file_merger.append(pdf,import_bookmarks=False)     # 合并pdf文件


file_merger.write(r"合并文件.pdf")

# 合并的时候，pdf_lst 是根据文件的名称来排序生成，如果对于pdf文件合成顺序有要求，建议吧文件按照期望的合成顺序编号1 2 3这样，方便一些