from win32com.client import constants, gencache
import os
import fitz
import tkinter as tk
from tkinter import filedialog
import msvcrt

pdf_dir = []
Word_files = []


def get_file():
    root = tk.Tk()
    root.withdraw()
    # 想要选中文件的，可以使用这个方法直接获取文件路径
    dir_path = filedialog.askdirectory(initialdir='./')
    # print("dir_path:"+dir_path)
    docunames = os.listdir(dir_path)

    for file in docunames:
        print("file:" + file)
        # 找出所有后缀为doc或者docx的文件
        if file.endswith(('.doc', '.docx')):
            Word_files.append(file)

    print(Word_files)

    for file in Word_files:
        file_path = os.path.abspath(dir_path + "/" + file)
        index = file_path.rindex('.')
        pdf_path = file_path[:index] + '.pdf'
        Word_to_Pdf(file_path, pdf_path)


# Word转pdf方法,第一个参数代表word文档路径，第二个参数代表pdf文档路径
def Word_to_Pdf(Word_path, Pdf_path):
    print("Word_path:" + Word_path)
    print("Pdf_path:" + Pdf_path)
    word = gencache.EnsureDispatch('Word.Application')
    doc = word.Documents.Open(Word_path)
    # 转换方法
    doc.ExportAsFixedFormat(Pdf_path, constants.wdExportFormatPDF)
    # word.Quit()


if __name__ == '__main__':
    get_file()
