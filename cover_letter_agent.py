import os
import openai
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

_ = load_dotenv(find_dotenv()) # read local .env file
api_key = os.environ['OPENAI_API_KEY']

# Create OpenAI client
client = OpenAI(api_key=api_key)

llm_model = "gpt-3.5-turbo"

def get_completion(prompt, model=llm_model):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content

print(get_completion("Write me cover letter with 1000 words for a Django developer posion at IBM, be creative"))


customer_skills = """
Python, CSS, HTML, DOCKER, Node JS, React JS, Machine Learning, JavaScript, CI/CD, SQL, MongoDB
"""


style = """American English \
respectful tone, creative, like a story, strong
"""

prompt = f"""Write me cover letter with 500 words for a Django developer posion at IBM, be creative\
into a style that is {style}.
with skills: ```{customer_skills}```
"""


response = get_completion(prompt)

print(response)


# text by an LLM, use temperature = 0.0
chat = ChatOpenAI(temperature=0.0, model=llm_model)

template_string = """Translate the text \
that is delimited by triple backticks \
into a style that is {style}. \
text: ```text```
"""

prompt_template = ChatPromptTemplate.from_template(template_string)

print(prompt_template.messages[0].prompt)