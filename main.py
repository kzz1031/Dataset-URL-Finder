from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import TextLoader
import os
import sys
from process_pdf_url import download_and_process_pdf

api_key = ''
with open('apikey.txt', 'r') as f:
    api_key = f.read().strip()
if not api_key:
    print("Deepseek API key not found. Please set your OpenAI API key in apikey.txt.")
    sys.exit(1)

chat = ChatOpenAI(
    model='deepseek-chat',
    openai_api_key=api_key,
    openai_api_base='https://api.deepseek.com',
    max_tokens=1024
)

def process_url_to_md(url):
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
        print("Usage: python main.py <pdf_url>")
        sys.exit(1)
    
    url = sys.argv[1]
    md_content = process_url_to_md(url)
    ###TO DO###
    if md_content:
        template_string = """分析以下文档内容，并提供一个简洁的摘要：
        
        文档内容: ```{text}```
        """
        
        prompt_template = ChatPromptTemplate.from_template(template_string)
        messages = prompt_template.format_messages(text=md_content)
        
        # 调用LLM处理文档内容
        response = chat.invoke(messages, temperature=0)
        print("\n文档摘要:")
        print(response.content)
    else:
        print("process_url_to_md failed")

