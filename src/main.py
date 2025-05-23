import os
import sys
from pdfurl2md import process_pdf_url_to_md
from urldigger import dig_urls_from_text, dig_context_of_urls
from urlprober import verify_urls
import json

def saveJson(filePath: str, datas: list) -> None:
    """保存数据到JSON文件"""
    assert isinstance(datas, list), "datas should be a list"
    directory = os.path.dirname(filePath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    try:
        with open(filePath, 'w', encoding='utf-8') as f:
            json.dump(datas, f, indent=4, ensure_ascii=False)
    except IOError as e:
        print(f"Error writing to file {filePath}: {e}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python src/main.py <pdf_url>")
        sys.exit(1)
    
    pdf_url = sys.argv[1]
    md_content = process_pdf_url_to_md(pdf_url)
    urls = dig_urls_from_text(md_content)
    url_context_dict = dig_context_of_urls(md_content, urls)
    
    # 验证URL并保存结果
    verified_urls = verify_urls(urls, url_context_dict)
    saveJson("output/dataset_urls.json", verified_urls)
    print(f"Found {len(verified_urls)} valid dataset URLs. Results saved to output/dataset_urls.json")


if __name__ == "__main__":
    main()
