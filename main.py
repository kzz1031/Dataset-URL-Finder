import os
import sys
from src.pdfurl2md import process_pdf_file_to_md
from src.urldigger import gather_texts, dig_urls_from_text, dig_context_of_urls
from src.urlprober import verify_urls
from src.main import saveJson

def determine_source_type(url: str) -> str:
    """根据URL确定数据源类型"""
    url_lower = url.lower()
    
    if 'github.com' in url_lower or 'githubusercontent.com' in url_lower:
        return 'github'
    elif 'huggingface.co' in url_lower or 'hf.co' in url_lower:
        return 'huggingface'
    elif 'kaggle.com' in url_lower:
        return 'kaggle'
    elif 'paperswithcode.com' in url_lower:
        return 'papers_with_code'
    elif 'openml.org' in url_lower:
        return 'openml'
    elif 'tensorflow.org' in url_lower or 'tfds' in url_lower:
        return 'tensorflow_datasets'
    elif 'pytorch.org' in url_lower or 'torchvision' in url_lower:
        return 'pytorch_datasets'
    elif 'dvc.org' in url_lower:
        return 'dvc'
    elif 'wandb.ai' in url_lower or 'wandb.com' in url_lower:
        return 'wandb'
    elif 'codalab.org' in url_lower:
        return 'codalab'
    elif 'nlp.stanford.edu' in url_lower:
        return 'stanford_nlp'
    elif 'cs.nyu.edu' in url_lower or 'cilvr.nyu.edu' in url_lower:
        return 'nyu'
    elif 'ai.google' in url_lower or 'research.google' in url_lower:
        return 'google_research'
    elif 'microsoft.com' in url_lower and ('research' in url_lower or 'dataset' in url_lower):
        return 'microsoft_research'
    elif 'facebook.com' in url_lower or 'meta.com' in url_lower:
        return 'meta_research'
    elif 'openai.com' in url_lower:
        return 'openai'
    elif 'anthropic.com' in url_lower:
        return 'anthropic'
    elif 'drive.google.com' in url_lower or 'docs.google.com' in url_lower:
        return 'google_drive'
    elif 'dropbox.com' in url_lower:
        return 'dropbox'
    elif 'zenodo.org' in url_lower:
        return 'zenodo'
    elif 'figshare.com' in url_lower:
        return 'figshare'
    elif 'arxiv.org' in url_lower:
        return 'arxiv'
    elif 'ieee.org' in url_lower or 'ieeexplore.ieee.org' in url_lower:
        return 'ieee'
    elif 'acm.org' in url_lower or 'dl.acm.org' in url_lower:
        return 'acm'
    elif 'springer.com' in url_lower or 'link.springer.com' in url_lower:
        return 'springer'
    elif 'elsevier.com' in url_lower or 'sciencedirect.com' in url_lower:
        return 'elsevier'
    elif 'onedrive.live.com' in url_lower or '1drv.ms' in url_lower:
        return 'onedrive'
    elif 'box.com' in url_lower:
        return 'box'
    elif 'mega.nz' in url_lower:
        return 'mega'
    elif 'amazonaws.com' in url_lower or 's3.' in url_lower:
        return 'aws_s3'
    elif 'storage.googleapis.com' in url_lower:
        return 'google_cloud'
    elif 'azure' in url_lower:
        return 'azure'
    else:
        return 'other'

def process_pdf_directory(pdf_dir: str, output_dir: str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    
    all_results = {}
    
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
            
            verified_urls = verify_urls(urls, url_context_dict)
            
            paper_results = {}
            for url_info in verified_urls:
                url = url_info.get('url', '')
                dataset_name = url.split('/')[-1].split('.')[0]
                
                description = ""
                if 'llm_details' in url_info and 'details' in url_info['llm_details']:
                    llm_details = url_info['llm_details']['details']
                    if 'explanation' in llm_details:
                        description = llm_details['explanation']
                
                source_type = determine_source_type(url)
    
                
                clean_url = url.rstrip(',')
                if not clean_url.startswith(('http://', 'https://')):
                    clean_url = 'https://' + clean_url
                
                paper_results[dataset_name] = [
                    source_type,
                    clean_url,
                    description
                ]
            
            all_results[pdf_name] = paper_results
            
            print(f"finish: {pdf_file}, find {len(verified_urls)} valid URL")
            
        except Exception as e:
            print(f"处理 {pdf_file} 时出错: {str(e)}")
            continue
    
    final_output_file = os.path.join(output_dir, "all_datasets.json")
    saveJson(final_output_file, all_results)
    print(f"所有结果已保存到: {final_output_file}")

def main():
    # 设置输入和输出目录
    pdf_dir = "课程作业论文"
    output_dir = "src/output"

    process_pdf_directory(pdf_dir, output_dir)

if __name__ == "__main__":
    main()
