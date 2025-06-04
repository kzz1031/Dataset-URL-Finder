import sys
import os
# from langchain_openai import ChatOpenAI
# from langchain.prompts import ChatPromptTemplate
# from langchain_community.document_loaders import TextLoader

# 获取当前文件所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
apikey_path = os.path.join(current_dir, 'apikey.txt')

api_key = ''
with open(apikey_path, 'r') as f:
    api_key = f.read().strip()
if not api_key:
    print("API key not found. Please set your OpenAI API key in src/apikey.txt.")
    sys.exit(1)

# chat_inst = ChatOpenAI(
#     model='deepseek-chat',
#     openai_api_key=api_key,
#     openai_api_base='https://api.deepseek.com',
#     max_tokens=10000
# )

import json
import requests

def qwen_by_api(prompt, engine_name = "chatgpt-4o-latest"):
    global api_key
    if "#" in engine_name:
        temp = engine_name.split("#")
        engine_name = temp[0]
        temperature = float(temp[1])
        params = {
            "messages": [{"role": "user", "content": prompt}],
            "model": engine_name,
            "temperature": temperature,
        }
    else:
        params = {
            "messages": [{"role": "user", "content": prompt}],
            "model": engine_name,
        }
    headers = {
        "Authorization": "Bearer " + api_key,
    }
    response = requests.post(
        "https://aigptx.top/v1/chat/completions",# 此处为中转地址，不要改动
        headers=headers,
        json=params,
        stream=False,
    )
    res = response.json()
    # print(res)
    message = res["choices"][0]["message"]["content"]
    usage = res["usage"]
    # print("message:"+message)
    # print(usage)
    return message

class ChatManager:
    def invoke(self, prompt):
        try:
            response = qwen_by_api(prompt)
            return response
        except Exception as e:
            print(f"Error invoking chat model: {e}")
            return None, None

chat_inst = ChatManager()
