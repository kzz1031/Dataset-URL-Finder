import sys
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader

# 获取当前文件所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
apikey_path = os.path.join(current_dir, 'apikey.txt')

api_key = ''
with open(apikey_path, 'r') as f:
    api_key = f.read().strip()
if not api_key:
    print("Deepseek API key not found. Please set your OpenAI API key in apikey.txt.")
    sys.exit(1)

chat_inst = ChatOpenAI(
    model='deepseek-chat',
    openai_api_key=api_key,
    openai_api_base='https://api.deepseek.com',
    max_tokens=10000
)

