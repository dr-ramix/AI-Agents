import os
import openai
from dotenv import load_dotenv, find_dotenv
from langchain.prompts import ChatPromptTemplate
from openai import OpenAI
#Chains
from langchain.chains import LLMChain

_ = load_dotenv(find_dotenv()) # read local .env file
api_key = os.environ['OPENAI_API_KEY']

# Create OpenAI client
client = OpenAI(api_key=api_key)

# Define the LLM model
llm_model = "gpt-3.5-turbo"

#create a model for the LLM
llm = ChatOpenAI(temperature=0.9, model=llm_model)

prompt = ChatPromptTemplate.from_template(
    "What is the best name to describe \
    a company that makes {product}?"
)

chain = LLMChain(llm=llm, prompt=prompt)

product = "Tablet for kids"
print(chain.run(product))