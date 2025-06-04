import os
import sys
from .pdfurl2md import process_pdf_url_to_md
from .pdfurl2md import process_pdf_file_to_md
from .urldigger import dig_urls_from_text, dig_context_of_urls, gather_texts
from .urlprober import verify_urls
import json

def saveJson(filePath: str, datas) -> None:
    assert isinstance(datas, (list, dict)), "datas should be a list or dict"
    directory = os.path.dirname(filePath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    try:
        with open(filePath, 'w', encoding='utf-8') as f:
            json.dump(datas, f, indent=4, ensure_ascii=False)
    except IOError as e:
        print(f"Error writing to file {filePath}: {e}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python src/main.py <mode> <input> [language]")
        print("Modes: url - Process a PDF from a URL")
        print("       file - Process a local PDF file")
        sys.exit(1)
    
    mode = sys.argv[1]
    input_path = sys.argv[2]
    lang = sys.argv[3] if len(sys.argv) > 3 else 'en'  # Default to English
    
    if mode == "url":
        outdir, mdname = process_pdf_url_to_md(input_path, lang)
    elif mode == "file":
        outdir, mdname = process_pdf_file_to_md(input_path, lang)
    else:
        print("Invalid mode. Use 'url' or 'file'.")
        sys.exit(1)
    
    texts = gather_texts(outdir, mdname)
    urls = dig_urls_from_text(texts)
    url_context_dict = dig_context_of_urls(texts, urls)
    
    # 验证URL并保存结果
    verified_urls = verify_urls(urls, url_context_dict)
    saveJson("output/dataset_urls.json", verified_urls)
    print(f"Found {len(verified_urls)} valid dataset URLs. Results saved to output/dataset_urls.json")


if __name__ == "__main__":
    main()
