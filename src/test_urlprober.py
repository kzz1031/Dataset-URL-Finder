import sys
import os
import traceback
import unittest
from unittest.mock import patch, MagicMock

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.urlprober import verify_urls, check_url_with_llm, check_url_accessibility

def test_single_url():
    try:
        print("\n=== Starting Single URL Test ===")
        # 测试单个URL
        test_url = "https://huggingface.co/datasets/mnist"  # 这是一个示例数据集URL
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
                    print(f"Reason: {result['reason']}")
                    print(f"LLM Details: {result['llm_details']}")
            else:
                print("No results returned from verify_urls")
        except Exception as e:
            print(f"Error in complete verification: {str(e)}")
            print("Stack trace:")
            traceback.print_exc()
            
    except Exception as e:
        print(f"Unexpected error in test_single_url: {str(e)}")
        print("Stack trace:")
        traceback.print_exc()

def test_multiple_urls():
    try:
        print("\n=== Starting Multiple URLs Test ===")
        # 测试多个URL
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
                    print(f"Reason: {result['reason']}")
                    print(f"LLM Details: {result['llm_details']}")
            else:
                print("No results returned from verify_urls")
        except Exception as e:
            print(f"Error in multiple URLs verification: {str(e)}")
            print("Stack trace:")
            traceback.print_exc()
            
    except Exception as e:
        print(f"Unexpected error in test_multiple_urls: {str(e)}")
        print("Stack trace:")
        traceback.print_exc()

class TestURLProber(unittest.TestCase):
    def setUp(self):
        self.test_url = "https://example.com/dataset"
        self.test_context = "This dataset is used for testing purposes."

    @patch('src.urlprober.chat_inst')
    def test_check_url_with_llm(self, mock_chat):
        # 测试正常情况
        mock_response = MagicMock()
        mock_response.content = "Score: 4\nExplanation: This is a clear dataset URL with proper context."
        mock_chat.invoke.return_value = mock_response
        
        score, details = check_url_with_llm(self.test_url, self.test_context)
        self.assertEqual(score, 4)
        self.assertEqual(details["status"], "success")
        self.assertIn("explanation", details["details"])
        
        # 测试错误情况
        mock_chat.invoke.side_effect = Exception("API Error")
        score, details = check_url_with_llm(self.test_url, self.test_context)
        self.assertEqual(score, 0)
        self.assertEqual(details["status"], "error")

    @patch('requests.get')
    @patch('requests.head')
    def test_check_url_accessibility(self, mock_head, mock_get):
        # 测试正常情况
        mock_head.return_value.status_code = 200
        mock_get.return_value.text = "<html>Dataset download page</html>"
        mock_get.return_value.status_code = 200
        
        score, details = check_url_accessibility(self.test_url)
        self.assertGreater(score, 0)
        self.assertEqual(details["status"], "success")
        
        # 测试Cloudflare保护
        mock_get.return_value.text = "<html>Cloudflare protection</html>"
        score, details = check_url_accessibility(self.test_url)
        self.assertEqual(score, 0)
        self.assertEqual(details["status"], "cloudflare")
        self.assertTrue(details["requires_manual_check"])
        
        # 测试404错误
        mock_head.return_value.status_code = 404
        score, details = check_url_accessibility(self.test_url)
        self.assertEqual(score, 0)
        self.assertEqual(details["status"], "not_found")
        
        # 测试403错误
        mock_head.return_value.status_code = 403
        score, details = check_url_accessibility(self.test_url)
        self.assertEqual(score, 0)
        self.assertEqual(details["status"], "restricted")

    @patch('src.urlprober.check_url_with_llm')
    @patch('src.urlprober.check_url_accessibility')
    def test_verify_urls(self, mock_accessibility, mock_llm):
        # 设置模拟返回值
        mock_llm.return_value = (4, {
            "status": "success",
            "message": "LLM analysis completed",
            "details": {"score": 4, "explanation": "Good dataset"}
        })
        mock_accessibility.return_value = (3, {
            "status": "success",
            "message": "URL is accessible",
            "details": {
                "url_score": 1.5,
                "content_score": 1.5,
                "url_details": ["URL matches github pattern"],
                "content_details": ["Page contains dataset information"]
            }
        })
        
        urls = [self.test_url]
        url_context_dict = {self.test_url: [self.test_context]}
        
        results = verify_urls(urls, url_context_dict, threshold=6)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["url"], self.test_url)
        self.assertEqual(results[0]["score"], 7)  # 4 + 3
        self.assertIn("llm_details", results[0])
        self.assertIn("reason", results[0])

if __name__ == "__main__":
    print("=== Starting URL Prober Tests ===")
    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")
    
    try:
        # 运行单个URL测试
        test_single_url()
        
        # 运行多个URL测试
        test_multiple_urls()
        
        # 运行单元测试
        unittest.main()
        
    except Exception as e:
        print(f"Unexpected error in main: {str(e)}")
        print("Stack trace:")
        traceback.print_exc()
    
    print("\n=== Tests Completed ===")