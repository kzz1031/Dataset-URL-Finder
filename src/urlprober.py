import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from .chat_manager import chat_inst
from tqdm import tqdm
import time
import re
import sys
import os
import traceback

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

def clean_url(url):
    """清理URL，移除末尾的特殊字符和其他问题"""
    # 移除引号、逗号、点和其他特殊字符
    url = url.strip()
    url = url.strip('"').strip(',').strip('.')
    # 移除URL末尾的引号和逗号
    url = re.sub(r'[",]+$', '', url)
    # 移除URL中的转义字符
    url = url.replace('\\', '')
    # 移除URL末尾的特殊字符
    url = url.rstrip('/').rstrip('\\').rstrip('.').rstrip(':').rstrip('?')
    # 如果URL以点结尾，移除点
    if url.endswith('.'):
        url = url[:-1]
    return url

def check_url_accessibility(url):
    """检查URL的可访问性和内容，返回0-5的分数和详细信息"""
    try:
        # 清理URL
        url = clean_url(url)
        print("cleaned url:")
        print(url)
        
        # 1. 检查URL是否可访问
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0'
        }
        
        try:
            response = requests.head(url, allow_redirects=True, timeout=10, headers=headers)
        except requests.exceptions.RequestException as e:
            # 如果HEAD请求失败，尝试GET请求
            try:
                response = requests.get(url, timeout=10, headers=headers)
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
            'huggingface': {
                'keywords': ['huggingface.co/datasets'],
                'weight': 4.0,
                'context_required': False
            },
            'kaggle': {
                'keywords': ['kaggle.com/datasets'],
                'weight': 4.0,
                'context_required': False
            },
            'dataset': {
                'keywords': ['dataset', 'datasets', 'data set', 'data sets'],
                'weight': 3.5,
                'context_required': False
            },
            'download': {
                'keywords': ['download', 'download data', 'download dataset'],
                'weight': 3.0,
                'context_required': False
            },
            'github': {
                'keywords': ['github.com', 'dataset', 'data'],
                'weight': 2.5,
                'context_required': True
            },
            'government': {
                'keywords': ['census.gov', 'data.gov', 'nasa.gov/data'],
                'weight': 3.0,
                'context_required': True
            },
            'academic': {
                'keywords': ['.edu/data', '.edu/dataset', '.edu/benchmark'],
                'weight': 2.5,
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
                    url_score = max(url_score, info['weight'])  # 使用最高权重而不是累加
                    url_details.append(f"URL matches {platform} pattern")
        
        # 检查页面内容特征
        page_text = soup.get_text().lower()
        content_score = 0
        content_details = []
        
        # 数据集相关关键词及其权重
        content_indicators = {
            'dataset_mention': {
                'keywords': ['dataset', 'benchmark', 'data collection', 'data repository'],
                'weight': 3.0
            },
            'download_info': {
                'keywords': ['download data', 'data files', 'data repository', 'download dataset'],
                'weight': 2.5
            },
            'data_format': {
                'keywords': ['format', 'structure', 'schema', 'csv', 'json', 'parquet'],
                'weight': 1.5
            },
            'access_info': {
                'keywords': ['access restricted', 'login required', 'registration required'],
                'weight': 1.0
            }
        }
        
        for category, info in content_indicators.items():
            if any(keyword in page_text for keyword in info['keywords']):
                content_score = max(content_score, info['weight'])  # 使用最高权重而不是累加
                content_details.append(f"Page contains {category} information")
        
        # 计算基础分数
        base_score = min(max(url_score, content_score), 5)
        
        # 如果URL可以访问，给予额外加分
        accessibility_bonus = 2.0 if response.status_code == 200 else 0.0
        
        # 计算总分（基础分数 + 可访问性加分，但不超过5）
        total_score = min(base_score + accessibility_bonus, 5)
        
        return total_score, {
            "status": "success",
            "message": "URL is accessible",
            "details": {
                "url_score": url_score,
                "content_score": content_score,
                "accessibility_bonus": accessibility_bonus,
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

def normalize_url(url):
    """规范化URL格式"""
    # 移除引号和逗号
    url = url.strip('"').strip(',').strip()
    # 转换为小写
    url = url.lower()
    # 移除末尾的斜杠和点
    url = url.rstrip('/').rstrip('.')
    # 移除URL参数
    url = re.sub(r'\?.*$', '', url)
    # 移除锚点
    url = re.sub(r'#.*$', '', url)
    # 移除www前缀
    url = re.sub(r'^www\.', '', url)
    return url

def calculate_url_similarity(url1, url2):
    """计算两个URL的相似度"""
    # 规范化URL
    url1 = normalize_url(url1)
    url2 = normalize_url(url2)
    
    # 如果两个URL完全相同，直接返回1.0
    if url1 == url2:
        return 1.0
    
    # 计算最长公共子序列
    def lcs(s1, s2):
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]
    
    # 计算相似度
    common_length = lcs(url1, url2)
    max_length = max(len(url1), len(url2))
    similarity = common_length / max_length
    
    return similarity

def check_url_duplicate(url1, url2):
    """使用AI检查两个URL是否指向相同的内容"""
    prompt = f"""Given these two URLs:
URL1: {url1}
URL2: {url2}

Please determine if these URLs point to the same content or resource. Consider:
1. Are they different versions of the same page?
2. Do they redirect to the same destination?
3. Are they different formats of the same dataset?
4. Are they different mirrors of the same content?

Respond with just 'DUPLICATE' or 'DIFFERENT' and no other explanation."""

    try:
        response = chat_inst.invoke(prompt)
        response_text = response.content.strip().upper()
        return response_text == 'DUPLICATE'
    except Exception as e:
        print(f"Error checking URL duplicate: {e}")
        return False

def verify_urls(urls, url_context_dict=None, threshold=4, similarity_threshold=0.8):
    """
    验证URL列表，返回得分超过阈值的URL
    Args:
        urls: URL列表
        url_context_dict: URL到上下文的映射字典
        threshold: 分数阈值（0-10）
        similarity_threshold: URL相似度阈值（0-1）
    Returns:
        list: 包含验证通过的URL信息的列表，每个元素是一个字典
    """
    # URL黑名单关键词
    blacklist_keywords = [
        'doi.org',  # DOI链接
        'proceedings',  # 会议论文集
        'z-lib',  # Z-Library
        # 'arxiv.org',  # arXiv论文
        # 'springer.com',  # Springer出版社
        # 'ieee.org',  # IEEE
        # 'acm.org',  # ACM
        # 'sciencedirect.com',  # ScienceDirect
        # 'wiley.com',  # Wiley
        # 'tandfonline.com',  # Taylor & Francis
        # 'sage.com',  # SAGE
        # 'mdpi.com',  # MDPI
        # 'hindawi.com',  # Hindawi
        # 'frontiersin.org',  # Frontiers
        # 'researchgate.net',  # ResearchGate
        # 'scholar.google.com',  # Google Scholar
        # 'semanticscholar.org',  # Semantic Scholar
        # 'jstor.org',  # JSTOR
        # 'sci-hub',  # Sci-Hub
        # 'libgen',  # Library Genesis
        
    ]
    
    # 过滤掉黑名单中的URL
    filtered_urls = []
    blacklisted_urls = []
    for url in urls:
        url_lower = url.lower()
        if any(keyword in url_lower for keyword in blacklist_keywords):
            blacklisted_urls.append(url)
        else:
            filtered_urls.append(url)
    
    # if blacklisted_urls:
    #     print(f"Filtered out {len(blacklisted_urls)} blacklisted URLs:")
    #     for url in blacklisted_urls:
    #         print(f"- {url}")
    
    # print(len(filtered_urls))
    # 在验证前进行URL去重
    print("Checking for duplicate URLs...")
    
    # 首先规范化所有URL
    normalized_urls = [(url, normalize_url(url)) for url in filtered_urls]
    # print(normalized_urls)
    # 使用字典来存储规范化后的URL到原始URL的映射
    norm_to_orig = {}
    for orig_url, norm_url in normalized_urls:
        if norm_url not in norm_to_orig:
            norm_to_orig[norm_url] = orig_url
        else:
            # 如果发现完全相同的规范化URL，保留较短的原始URL
            # print(orig_url, norm_url)
            if len(orig_url) < len(norm_to_orig[norm_url]):
                norm_to_orig[norm_url] = orig_url
    
    # 获取去重后的URL列表
    unique_urls = list(norm_to_orig.values())
    
    print(f"Found {len(filtered_urls)} URLs, {len(unique_urls)} unique URLs after basic deduplication")
    # print(unique_urls)
    # 对剩余的URL进行相似度比较
    similar_pairs = []
    for i in range(len(unique_urls)):
        for j in range(i + 1, len(unique_urls)):
            # 只有当两个URL的域名相同时才进行相似度比较
            domain1 = urlparse(unique_urls[i]).netloc
            domain2 = urlparse(unique_urls[j]).netloc
            if domain1 == domain2:
                similarity = calculate_url_similarity(unique_urls[i], unique_urls[j])
                if similarity >= similarity_threshold:
                    similar_pairs.append((unique_urls[i], unique_urls[j], similarity))
    
    # 使用AI确认重复
    duplicates = set()
    if similar_pairs:
        print(f"Found {len(similar_pairs)} potential duplicate URL pairs, checking with AI...")
        for url1, url2, similarity in tqdm(similar_pairs):
            if check_url_duplicate(url1, url2):
                # 保留较短的URL
                duplicates.add(url1 if len(url1) > len(url2) else url2)
    
    # 构建最终的去重URL列表
    final_urls = [url for url in unique_urls if url not in duplicates]
    print(f"Final unique URLs after AI verification: {len(final_urls)}")
    
    verified_urls = []
    
    print("Verifying URLs...")
    for url in tqdm(final_urls):
        # 获取URL的上下文
        context = url_context_dict.get(url, [""])[0] if url_context_dict else ""
        
        # 步骤1：使用LLM检查
        llm_score, llm_details = check_url_with_llm(url, context)
        
        # 步骤2：检查URL可访问性
        access_score, access_details = check_url_accessibility(url)
        
        # 计算总分
        total_score = llm_score*1.2 + access_score*0.8

        print(f"URL: {url}, Total Score: {total_score}/10, LLM Score: {llm_score}/5, Access Score: {access_score}/5")
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
    
    # 按score降序排序verified_urls
    verified_urls.sort(key=lambda x: x['score'], reverse=True)
    
    return verified_urls

def test_urlprober():
    """运行URL验证系统的测试"""
    print("=== Starting URL Prober Tests ===")
    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")
    
    try:
        # 测试单个URL
        print("\n=== Starting Single URL Test ===")
        test_url = "https://huggingface.co/datasets/mnist"
        test_context = "The MNIST dataset is a large database of handwritten digits that is commonly used for training various image processing systems."
        
        print(f"\n1. Input Information:")
        print(f"URL: {test_url}")
        print(f"Context: {test_context}")
        
        # 测试LLM检查
        print("\n2. Starting LLM check...")
        try:
            llm_score, llm_details = check_url_with_llm(test_url, test_context)
            print(f"LLM Score: {llm_score}/5")
            print(f"LLM Details: {llm_details}")
        except Exception as e:
            print(f"Error in LLM check: {str(e)}")
            print("Stack trace:")
            traceback.print_exc()
        
        # 测试URL可访问性检查
        print("\n3. Starting URL accessibility check...")
        try:
            access_score, details = check_url_accessibility(test_url)
            print(f"Accessibility Score: {access_score}/5")
            print(f"Details: {details}")
        except Exception as e:
            print(f"Error in accessibility check: {str(e)}")
            print("Stack trace:")
            traceback.print_exc()
        
        # 测试完整验证
        print("\n4. Starting complete verification...")
        try:
            results = verify_urls([test_url], {test_url: [test_context]})
            print("\nFinal Results:")
            if results:
                for result in results:
                    print(f"\nURL: {result['url']}")
                    print(f"Total Score: {result['score']}/10")
                    print(f"LLM Score: {result['llm_score']}/5")
                    print(f"Access Score: {result['access_score']}/5")
                    print(f"Access Details: {result['access_details']}")
                    print(f"LLM Details: {result['llm_details']}")
            else:
                print("No results returned from verify_urls")
        except Exception as e:
            print(f"Error in complete verification: {str(e)}")
            print("Stack trace:")
            traceback.print_exc()
        
        # 测试多个URL
        print("\n=== Starting Multiple URLs Test ===")
        test_urls = [
            "https://huggingface.co/datasets/mnist",  # 数据集
            "https://www.cs.toronto.edu/~kriz/cifar.html",     # 数据集
            "https://github.com/kzz1031/Dataset-URL-Finder"   # 项目主页
        ]
        
        test_contexts = {
            test_urls[0]: ["The MNIST dataset is a large database of handwritten digits."],
            test_urls[1]: ["PyTorch is a machine learning framework."],
            test_urls[2]: ["MNIST dataset on Kaggle platform."]
        }
        
        print("\n1. Input Information:")
        print("URLs to test:")
        for url in test_urls:
            print(f"- {url}")
        
        print("\n2. Starting verification of multiple URLs...")
        try:
            results = verify_urls(test_urls, test_contexts, 0)
            
            print("\nResults for all URLs:")
            if results:
                for result in results:
                    print(f"\nURL: {result['url']}")
                    print(f"Total Score: {result['score']}/10")
                    print(f"LLM Score: {result['llm_score']}/5")
                    print(f"Access Score: {result['access_score']}/5")
                    print(f"Access Details: {result['access_details']}")
                    print(f"LLM Details: {result['llm_details']}")
            else:
                print("No results returned from verify_urls")
        except Exception as e:
            print(f"Error in multiple URLs verification: {str(e)}")
            print("Stack trace:")
            traceback.print_exc()
            
    except Exception as e:
        print(f"Unexpected error in tests: {str(e)}")
        print("Stack trace:")
        traceback.print_exc()
    
    print("\n=== Tests Completed ===")

if __name__ == "__main__":
    test_urlprober()
