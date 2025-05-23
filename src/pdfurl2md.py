import os
import sys
import requests
import subprocess


def download_and_process_pdf(url, output_dir):
    filename = url.split('/')[-1]
    if not filename.endswith('.pdf'):
        filename = f"{filename}.pdf"
    
    local_path = os.path.join(output_dir, filename)
    
    print(f"Download PDF from {url} to {local_path}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(local_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"Using magic-pdf to process {local_path}...")
    subprocess.run(['magic-pdf', '-p', local_path, '-o', output_dir, '-m', 'auto'])
    
    return local_path

def process_pdf_url_to_md(url):
    print(f"Processing URL: {url}")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    pdf_path = download_and_process_pdf(url, output_dir)
    md_filename = os.path.splitext(os.path.basename(pdf_path))[0]
    md_path = os.path.join(output_dir, md_filename, 'auto', md_filename + '.md') # change 'auto' if you use other method in process_pdf_url.py
    
    if not os.path.exists(md_path):
        print(f"MD file not found in: {md_path}")
        return None
    
    with open(md_path, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    return md_content

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process_pdf_url.py <url> <output_dir>")
        sys.exit(1)
    
    url = sys.argv[1]
    output_dir = 'output'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    local_path = download_and_process_pdf(url, output_dir)
    print(f"PDF downloaded and processed to {local_path}")
