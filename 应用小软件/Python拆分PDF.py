
from PyPDF2 import PdfFileReader, PdfFileWriter


# PDF文件分割
def split_pdf(pdf_path=''):
    try:
        # read_file = input("请输入要拆分的PDF名字(例如test.pdf): ")
        fp_read_file = open(pdf_path, 'rb')
        pdf_input = PdfFileReader(fp_read_file)  # 将要分割的PDF内容格式话
        page_count = pdf_input.getNumPages()  # 获取PDF页数
        print("该文件共有{}页".format(page_count))  # 打印页数
        # out_detail=input("请输入写有拆分规则的文件名（例如：rule.txt）: ")
        out_detail=r'split_pdf.txt'
        print(out_detail)
        with open(out_detail, 'r',True,'utf-8')as fp:
            print(out_detail)
            txt = fp.readlines()
            # print(txt)
            for detail in txt:  # 打开分割标准文件
                # print(type(detail))
                pages, write_file = detail.split()  # 空格分组
               #  write_file, write_ext = os.path.splitext(write_file)  # 用于返回文件名和扩展名元组
                pdf_file = f'{write_file}.pdf'
                print(pdf_file)
                # liststr=list(map(int, pages.split('-')))
                # print(type(liststr))
                start_page, end_page = list(map(int, pages.split('-')))  # 将字符串数组转换成整形数组
                start_page -= 1
                try:
                    print(f'开始分割{start_page}页-{end_page}页，保存为{pdf_file}......')
                    pdf_output = PdfFileWriter()  # 实例一个 PDF文件编写器
                    for i in range(start_page, end_page):
                        pdf_output.addPage(pdf_input.getPage(i))
                    with open(pdf_file, 'wb') as sub_fp:
                        pdf_output.write(sub_fp)
                    print(f'完成分割{start_page}页-{end_page}页，保存为{pdf_file}!')
                except IndexError:
                    print(f'分割页数超过了PDF的页数')
        # fp.close()
    except Exception as e:
        print(e)
    # finally:
    #     fp_read_file.close()



# def main():
#     fire.Fire(split_pdf)
#
# if __name__ == '__main__':
#     main()
if __name__ == '__main__':
    pdf_path = r'C:\Users\34426\Documents\WeChat Files\wxid_gr1sr01q93ug22\FileStorage\File\2022-11\市卫生健康委 市教育局关于印发深圳市2022年适龄女生人乳头瘤病毒（HPV）疫苗免费接种工作方案的通知.pdf'
    print("******************************说明***************************************")
    print("\t该程序需要两个文件，并添加到该路径下")
    print("\t一个是要拆分的PDF文档，另一个是拆分规则文档（用txt文档写）")
    print("\t拆分规则：页数范围与拆分后文档名，一个文档写一行。例如：")
    print("\t\t\t\t\t1-40 one\n\t\t\t\t\t41-85 two\n\t\t\t\t\t86-125 three")
    print("************************************************************************")

    split_pdf(pdf_path)
