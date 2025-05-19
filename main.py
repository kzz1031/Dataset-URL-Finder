from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import TextLoader
import os
import sys
from url2md import process_url_to_md

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

