import os
import sys
import requests
import subprocess
import fitz  # PyMuPDF
import hashlib
from .logger_config import setup_logger

logger = setup_logger(__name__)

def pdf2md(pdf_path, output_dir, lang='en'):
    logger.info(f"Converting PDF to markdown: {pdf_path}")
    logger.debug(f"Output directory: {output_dir}, Language: {lang}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filename = os.path.splitext(os.path.basename(pdf_path))[0]
    md_path = os.path.join(output_dir, filename + '.md')
    
    try:
        subprocess.run(['magic-pdf', '-p', pdf_path, '-o', output_dir, '-m', 'auto', '--lang', lang], check=True)
        logger.info(f"PDF conversion successful: {md_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"PDF conversion failed: {e}")
        raise
    
    return md_path

def download_and_process_pdf(url, output_dir, lang='en'):
    logger.info(f"Downloading PDF from URL: {url}")
    
    filename = url.split('/')[-1]
    if not filename.endswith('.pdf'):
        filename = f"{filename}.pdf"
        if len(filename) > 100:
            filename = filename[:100] + '.pdf'
            logger.warning(f"Filename truncated to: {filename}")
    
    local_path = os.path.join(output_dir, filename)
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"PDF downloaded successfully: {local_path}")
        
        pdf2md(local_path, output_dir, lang)
        return local_path
        
    except requests.RequestException as e:
        logger.error(f"Failed to download PDF from {url}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        raise

def process_pdf_url_to_md(url, lang='en'):
    logger.info(f"Processing URL: {url}")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    pdf_path = download_and_process_pdf(url, output_dir, lang)
    md_filename = os.path.splitext(os.path.basename(pdf_path))[0]
    md_path = os.path.join(output_dir, md_filename, 'auto', md_filename + '.md') # change 'auto' if you use other method in process_pdf_url.py
    
    if not os.path.exists(md_path):
        logger.error(f"MD file not found in: {md_path}")
        return None
    
    return os.path.join(output_dir, md_filename, 'auto'), md_filename

def process_pdf_file_to_md(pdf_path: str, lang: str) -> tuple:
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found: {pdf_path}")
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    logger.info(f"Processing local PDF file: {pdf_path}")
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pdf2md(pdf_path, output_dir, lang)
    
    md_filename = os.path.splitext(os.path.basename(pdf_path))[0]
    md_path = os.path.join(output_dir, md_filename, 'auto', md_filename + '.md')  
    
    if not os.path.exists(md_path):
        logger.error(f"Markdown file not found after conversion: {md_path}")
        return None
    
    result_dir = os.path.join(output_dir, md_filename, 'auto')
    logger.info(f"PDF processing complete: {result_dir}")
    
    return result_dir, md_filename

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process_pdf_url.py <url> <output_dir>")
        sys.exit(1)
    
    url = sys.argv[1]
    output_dir = 'output'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    pdf2md('output/C4NbtYnyQg.pdf', output_dir, 'en')
    # local_path = download_and_process_pdf(url, output_dir)
    # print(f"PDF downloaded and processed to {local_path}")
