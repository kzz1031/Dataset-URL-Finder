import os
import sys
from pdfurl2md import process_pdf_url_to_md
from urldigger import dig_urls_from_text, dig_context_of_urls
from urlprober import verify_urls


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <pdf_url>")
        sys.exit(1)
    
    pdf_url = sys.argv[1]
    md_content = process_pdf_url_to_md(pdf_url)
    urls = dig_urls_from_text(md_content)
    url_context_dict = dig_context_of_urls(md_content, urls)
    verified_urls = verify_urls(urls)
    print(verified_urls)


if __name__ == "__main__":
    main()
