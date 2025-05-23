import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from chat_manager import chat_inst
from tqdm import tqdm
import time
import re

def check_url_with_llm(url, context):
    """使用LLM检查URL是否为数据集，返回0-5的分数和详细信息"""
    prompt = f"""
    Given the following URL and its context from an academic paper:
    URL: {url}
    Context: {context}
    
    Please determine if this URL points to a dataset/benchmark that can be downloaded or accessed.
    Consider the following criteria:
    1. Is it a direct link to a dataset download page or official repository?
    2. Is it NOT a project homepage, blog, or general documentation?
    3. Is it specifically related to data/benchmarks mentioned in the paper?
    4. If the URL is no longer accessible (404, 403, etc.), does the context suggest it was a dataset?
    5. If the URL requires special access (like government websites), does the context indicate it's a dataset?
    
    Rate this URL from 0 to 5, where:
    0: Definitely not a dataset
    1: Very unlikely to be a dataset
    2: Possibly a dataset
    3: Likely a dataset
    4: Very likely a dataset
    5: Definitely a dataset
    
    Also provide a brief explanation for your rating.
    
    Respond in the following format:
    Score: [0-5]
    Explanation: [your explanation]
    """
    
    try:
        response = chat_inst.invoke(prompt)
        response_text = response.content.strip()
        
        # 解析响应
        score_match = re.search(r'Score:\s*(\d+)', response_text)
        explanation_match = re.search(r'Explanation:\s*(.*?)(?:\n|$)', response_text, re.DOTALL)
        
        if score_match and explanation_match:
            score = int(score_match.group(1))
            explanation = explanation_match.group(1).strip()
        else:
            # 如果无法解析格式，尝试直接获取数字
            score = int(response_text.split()[0])
            explanation = "No detailed explanation provided"
        
        return min(max(score, 0), 5), {
            "status": "success",
            "message": "LLM analysis completed",
            "details": {
                "score": score,
                "explanation": explanation,
                "raw_response": response_text
            }
        }
    except Exception as e:
        return 0, {
            "status": "error",
            "message": f"Error in LLM check: {str(e)}",
            "details": "Failed to analyze URL with LLM"
        }

def check_url_accessibility(url):
    """检查URL的可访问性和内容，返回0-5的分数和详细信息"""
    try:
        # 1. 检查URL是否可访问
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        try:
            response = requests.head(url, allow_redirects=True, timeout=10, headers=headers)
        except requests.exceptions.RequestException as e:
            return 0, {
                "status": "error",
                "message": f"Request failed: {str(e)}",
                "details": "URL is completely inaccessible"
            }
        
        # 检查特殊状态码
        if response.status_code == 403:
            return 0, {
                "status": "restricted",
                "message": "Access restricted (403)",
                "details": "This URL requires special access permissions (e.g., government website)"
            }
        elif response.status_code == 404:
            return 0, {
                "status": "not_found",
                "message": "URL not found (404)",
                "details": "This URL no longer exists or has been moved"
            }
        elif response.status_code != 200:
            return 0, {
                "status": "error",
                "message": f"URL not accessible (Status code: {response.status_code})",
                "details": "The URL is not accessible for technical reasons"
            }
        
        # 2. 获取页面内容
        try:
            response = requests.get(url, timeout=10, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
        except Exception as e:
            return 0, {
                "status": "error",
                "message": f"Error parsing page: {str(e)}",
                "details": "Could not parse the page content"
            }
        
        # 3. 检查Cloudflare保护
        if "cloudflare" in response.text.lower():
            return 0, {
                "status": "cloudflare",
                "message": "Cloudflare protected",
                "details": "This URL is protected by Cloudflare. Manual verification required.",
                "requires_manual_check": True
            }
        
        # 4. 检查页面特征
        dataset_indicators = {
            'download': {
                'keywords': ['download', 'dataset', 'data', 'benchmark'],
                'weight': 0.3,
                'context_required': True
            },
            'github': {
                'keywords': ['github.com', 'dataset', 'data'],
                'weight': 0.4,
                'context_required': True
            },
            'huggingface': {
                'keywords': ['huggingface.co/datasets'],
                'weight': 0.8,
                'context_required': False
            },
            'kaggle': {
                'keywords': ['kaggle.com/datasets'],
                'weight': 0.8,
                'context_required': False
            },
            'government': {
                'keywords': ['census.gov', 'data.gov', 'nasa.gov/data'],
                'weight': 0.6,
                'context_required': True
            },
            'academic': {
                'keywords': ['.edu/data', '.edu/dataset', '.edu/benchmark'],
                'weight': 0.7,
                'context_required': True
            }
        }
        
        # 检查URL特征
        url_lower = url.lower()
        url_score = 0
        url_details = []
        
        for platform, info in dataset_indicators.items():
            if any(keyword in url_lower for keyword in info['keywords']):
                if not info['context_required'] or any(keyword in url_lower for keyword in ['dataset', 'data', 'benchmark']):
                    url_score += info['weight']
                    url_details.append(f"URL matches {platform} pattern")
        
        # 检查页面内容特征
        page_text = soup.get_text().lower()
        content_score = 0
        content_details = []
        
        # 数据集相关关键词及其权重
        content_indicators = {
            'dataset_mention': {
                'keywords': ['dataset', 'benchmark', 'data collection'],
                'weight': 0.4
            },
            'download_info': {
                'keywords': ['download data', 'data files', 'data repository'],
                'weight': 0.3
            },
            'data_format': {
                'keywords': ['format', 'structure', 'schema', 'csv', 'json', 'parquet'],
                'weight': 0.2
            },
            'access_info': {
                'keywords': ['access restricted', 'login required', 'registration required'],
                'weight': 0.1
            }
        }
        
        for category, info in content_indicators.items():
            if any(keyword in page_text for keyword in info['keywords']):
                content_score += info['weight']
                content_details.append(f"Page contains {category} information")
        
        # 计算总分
        total_score = min(url_score + content_score, 5)
        
        return total_score, {
            "status": "success",
            "message": "URL is accessible",
            "details": {
                "url_score": url_score,
                "content_score": content_score,
                "url_details": url_details,
                "content_details": content_details
            }
        }
        
    except Exception as e:
        return 0, {
            "status": "error",
            "message": f"Error: {str(e)}",
            "details": "An unexpected error occurred during URL checking"
        }

def verify_urls(urls, url_context_dict=None, threshold=6):
    """
    验证URL列表，返回得分超过阈值的URL
    Args:
        urls: URL列表
        url_context_dict: URL到上下文的映射字典
        threshold: 分数阈值（0-10）
    Returns:
        list: 包含验证通过的URL信息的列表，每个元素是一个字典
    """
    verified_urls = []
    
    print("Verifying URLs...")
    for url in tqdm(urls):
        # 获取URL的上下文
        context = url_context_dict.get(url, [""])[0] if url_context_dict else ""
        
        # 步骤1：使用LLM检查
        llm_score, llm_details = check_url_with_llm(url, context)
        
        # 步骤2：检查URL可访问性
        access_score, access_details = check_url_accessibility(url)
        
        # 计算总分
        total_score = llm_score + access_score
        
        # 如果总分超过阈值，添加到已验证URL列表
        if total_score >= threshold:
            verified_urls.append({
                'url': url,
                'score': total_score,
                'llm_score': llm_score,
                'access_score': access_score,
                'llm_details': llm_details,
                'access_details': access_details,
                'context': context
            })
        
        # 添加延迟以避免请求过快
        time.sleep(1)
    
    # 去重处理
    unique_urls = []
    seen_urls = set()
    for url_info in verified_urls:
        if url_info['url'] not in seen_urls:
            seen_urls.add(url_info['url'])
            unique_urls.append(url_info)
    
    return unique_urls
