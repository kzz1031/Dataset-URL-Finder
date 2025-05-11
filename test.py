from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
api_key = 'sk-7249b1c247fb48808f8e8bb61f897919'
chat = ChatOpenAI(
    model='deepseek-chat',
    openai_api_key=api_key,
    openai_api_base='https://api.deepseek.com',
    max_tokens=1024
)


template_string = """Translate the text \
that is delimited by triple backticks \
into a style that is {style}. \
text: ```{text}```
"""


prompt_template = ChatPromptTemplate.from_template(template_string)
customer_style = """American English \
in a calm and respectful tone
"""
customer_email = """
Arrr, I be fuming that me blender lid \
flew off and splattered me kitchen walls \
with smoothie! And to make matters worse, \
the warranty don't cover the cost of \
cleaning up me kitchen. I need yer help \
right now, matey!
"""
customer_messages = prompt_template.format_messages(
                    style=customer_style,
                    text=customer_email)
# Call the LLM to translate to the style of the customer message
# Reference: chat = ChatOpenAI(temperature=0.0)
customer_response = chat.invoke(customer_messages, temperature=0)
print(customer_response.content)


service_reply = """Hey there customer, \
the warranty does not cover \
cleaning expenses for your kitchen \
because it's your fault that \
you misused your blender \
by forgetting to put the lid on before \
starting the blender. \
Tough luck! See ya!
"""

service_style_pirate = """\
a polite tone \
that speaks in English Pirate\
"""

service_messages = prompt_template.format_messages(
    style=service_style_pirate,
    text=service_reply)

service_response = chat.invoke(service_messages, temperature=0)
print(service_response.content)
