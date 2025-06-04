import os
import sys
from src.pdfurl2md import process_pdf_file_to_md
from src.urldigger import gather_texts, dig_urls_from_text, dig_context_of_urls
from src.urlprober import verify_urls, clean_and_deduplicate_urls
from src.main import saveJson
from src.logger_config import setup_logger
from pprint import pprint
import json

logger = setup_logger(__name__)

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

def save_urls_to_text(pdf_name: str, urls: list, output_dir: str):
    """将verified_urls保存到文本文件"""
    urls_dir = os.path.join(output_dir, "urls_text")
    if not os.path.exists(urls_dir):
        os.makedirs(urls_dir)
    
    text_file_path = os.path.join(urls_dir, f"{pdf_name}_urls.txt")
    
    with open(text_file_path, 'w', encoding='utf-8') as f:
        for i, url in enumerate(urls, 1):
            f.write(f"{url}\n")
    logger.info(f"Saved {len(urls)} URLs to: {text_file_path}")

def check_urls_file_exists(pdf_name: str, output_dir: str) -> bool:
    """检查该论文的URLs文本文件是否已存在"""
    urls_dir = os.path.join(output_dir, "urls_text")
    text_file_path = os.path.join(urls_dir, f"{pdf_name}_urls.txt")
    return os.path.exists(text_file_path)

def check_individual_result_exists(pdf_name: str, output_dir: str) -> bool:
    """检查该论文的个人结果JSON文件是否已存在"""
    individual_results_dir = os.path.join(output_dir, "individual_results")
    result_file_path = os.path.join(individual_results_dir, f"{pdf_name}_datasets.json")
    return os.path.exists(result_file_path)

def load_individual_result(pdf_name: str, output_dir: str) -> dict:
    """加载已存在的个人结果"""
    individual_results_dir = os.path.join(output_dir, "individual_results")
    result_file_path = os.path.join(individual_results_dir, f"{pdf_name}_datasets.json")
    
    try:
        with open(result_file_path, 'r', encoding='utf-8') as f:
            result = json.load(f)
            return result.get('datasets', {})
    except Exception as e:
        logger.error(f"Error loading individual result for {pdf_name}: {e}")
        return {}

def process_pdf_directory(pdf_dir: str, output_dir: str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    all_results = {}
    
    # Create individual results directory
    individual_results_dir = os.path.join(output_dir, "individual_results")
    if not os.path.exists(individual_results_dir):
        os.makedirs(individual_results_dir)
    
    skip = 0
    outdir = ''
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        pdf_name = os.path.splitext(pdf_file)[0]
        
        if check_individual_result_exists(pdf_name, output_dir):
            logger.info(f"Individual result already exists for {pdf_file}, loading existing data")
            paper_results = load_individual_result(pdf_name, output_dir)
            all_results[pdf_name] = paper_results
            logger.info(f"Loaded {len(paper_results)} datasets from existing result for {pdf_file}")
            continue
        
        if check_urls_file_exists(pdf_name, output_dir):
            logger.info(f"URLs file already exists for {pdf_file}, skipping URL extraction")
            continue
        
        output_pdf_dir = os.path.join(output_dir, pdf_name)
        if os.path.exists(output_pdf_dir):
            logger.info(f"Skipping PDF processing for {pdf_file} (already processed)")
            skip = 1
            outdir = output_pdf_dir + '/auto'
            mdname = pdf_name
            
        logger.info(f"Processing: {pdf_file}")
        try:
            if not skip:
                logger.info(f"Converting PDF to markdown: {pdf_file}")
                outdir, mdname = process_pdf_file_to_md(pdf_path, 'en')
            if skip:
                logger.debug("Skipping PDF conversion (already done)")
                skip = 0

            deduplicated_urls = []
            url_context_dict = {}
            texts = gather_texts(outdir, mdname)
            
            if not check_urls_file_exists(pdf_name, '.'):
                logger.info(f"Extracting URLs from text content: {outdir}")
                urls = dig_urls_from_text(texts)
                url_context_dict = dig_context_of_urls(texts, urls)
                deduplicated_urls, url_context_dict = clean_and_deduplicate_urls(urls, url_context_dict)
                save_urls_to_text(pdf_name, deduplicated_urls, '.')
            else:
                logger.info("Loading URLs from existing file")
                with open(os.path.join('.', 'urls_text', f"{pdf_name}_urls.txt"), 'r', encoding='utf-8') as f:
                    deduplicated_urls = [line.strip() for line in f if line.strip()]
                if deduplicated_urls:
                    url_context_dict = dig_context_of_urls(texts, deduplicated_urls)
            
            logger.info(f"Verifying {len(deduplicated_urls)} URLs")
            verified_urls = verify_urls(deduplicated_urls, url_context_dict)
            
            # 保存URLs到文本文件
            
            paper_results = {}
            for url_info in verified_urls:
                url = url_info.get('url', '')
                dataset_name = url.split('/')[-1].split('.')[0];
                
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
            
            # Save individual PDF results to separate JSON file
            individual_result_file = os.path.join(individual_results_dir, f"{pdf_name}_datasets.json")
            individual_result = {
                "pdf_name": pdf_name,
                "total_urls_found": len(verified_urls),
                "datasets": paper_results,
                "verified_urls_details": verified_urls  # Include detailed verification info
            }
            saveJson(individual_result_file, individual_result)
            logger.info(f"Individual results saved to: {individual_result_file}")
            
            all_results[pdf_name] = paper_results
            
            logger.info(f"Completed {pdf_file}: found {len(verified_urls)} valid URLs")
            
        except Exception as e:
            logger.error(f"Error processing {pdf_file}: {str(e)}", exc_info=True)
            continue
    
    final_output_file = os.path.join(output_dir, "all_datasets.json")
    saveJson(final_output_file, all_results)
    logger.info(f"All results saved to: {final_output_file}")
    logger.info(f"Individual results saved in: {individual_results_dir}")

def main():
    logger.info("Starting Dataset URL Finder")
    pdf_dir = "课程作业论文"
    output_dir = "src/output"
    
    logger.info(f"Input directory: {pdf_dir}")
    logger.info(f"Output directory: {output_dir}")

    process_pdf_directory(pdf_dir, output_dir)
    logger.info("Processing complete")

if __name__ == "__main__":
    main()
