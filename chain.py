import os
import openai
from dotenv import load_dotenv, find_dotenv
from langchain.prompts import ChatPromptTemplate
from openai import OpenAI
from langchain_openai import ChatOpenAI
#Chains
from langchain.chains import LLMChain

#Sequential chains
from langchain.chains import SequentialChain

_ = load_dotenv(find_dotenv()) # read local .env file
api_key = os.environ['OPENAI_API_KEY']

# Create OpenAI client
client = OpenAI(api_key=api_key)

# Define the LLM model
llm_model = "gpt-3.5-turbo"

#create a model for the LLM
llm = ChatOpenAI(temperature=0.9, model=llm_model)

prompt = ChatPromptTemplate.from_template(
    "Write an interview question for  \
    a job with title: {job}?"
)
#Definiton of the chain
chain = LLMChain(llm=llm, prompt=prompt)

job = "Machine Learning Engineer"
print(chain.run(job))