import os
import sys
from src.pdfurl2md import process_pdf_file_to_md
from src.urldigger import gather_texts, dig_urls_from_text, dig_context_of_urls
from src.urlprober import verify_urls
from src.main import saveJson

def process_pdf_directory(pdf_dir: str, output_dir: str):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    
    skip = 0
    outdir = ''

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        pdf_name = os.path.splitext(pdf_file)[0]
        
        # 检查是否已经在output目录中处理过
        output_pdf_dir = os.path.join(output_dir, pdf_name)
        if os.path.exists(output_pdf_dir):
            print(f"skip the file: {pdf_file}")
            skip = 1
            outdir = output_pdf_dir + '/auto'
            mdname = pdf_name
            
        print(f"processing: {pdf_file}")
        try:

            if not skip:
                outdir, mdname = process_pdf_file_to_md(pdf_path, 'en')
            if skip:
                print("skipping...")
                skip = 0

            print("gathering text from ", outdir)
            texts = gather_texts(outdir, mdname)
            urls = dig_urls_from_text(texts)
            url_context_dict = dig_context_of_urls(texts, urls)
            
            # 验证URL并保存结果
            verified_urls = verify_urls(urls, url_context_dict)
            
            # 保存结果到对应的输出目录
            output_file = os.path.join(output_dir, f"{pdf_name}_urls.json")
            saveJson(output_file, verified_urls)
            
            print(f"finish: {pdf_file}, find {len(verified_urls)} valid URL")
            
        except Exception as e:
            print(f"处理 {pdf_file} 时出错: {str(e)}")
            continue

def main():
    # 设置输入和输出目录
    pdf_dir = "课程作业论文"
    output_dir = "src/output"

    process_pdf_directory(pdf_dir, output_dir)

if __name__ == "__main__":
    main() 