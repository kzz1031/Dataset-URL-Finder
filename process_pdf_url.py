import os
import sys
import requests
import subprocess
### You need to install magic-pdf first ###
def download_and_process_pdf(url, output_dir):
    # Extract filename from URL
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

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <url> <output_dir>")
        sys.exit(1)
    
    url = sys.argv[1]
    output_dir = sys.argv[2]
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    local_path = download_and_process_pdf(url, output_dir)
    print(f"PDF downloaded and processed to {local_path}")