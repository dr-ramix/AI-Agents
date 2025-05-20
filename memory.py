import os
import openai
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain

from langchain.memory import ConversationBufferMemory

_ = load_dotenv(find_dotenv()) # read local .env file
api_key = os.environ['OPENAI_API_KEY']

# Create OpenAI client
client = OpenAI(api_key=api_key)

llm_model = "gpt-3.5-turbo"

llm = ChatOpenAI(temperature=0.0, model=llm_model)
memory = ConversationBufferMemory()
converstation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

converstation.predict(input="Hello, how are you?") # this will be the first message
converstation.predict(input="What is your name?") # this will be the second message
converstation.predict(input="What is 1 + 1") # this will be the third message
#print(memory.buffer)

memory.load_memory_variables({}) # this will load the memory variables
#print(memory.load_memory_variables({}))
memory = ConversationBufferMemory()
memory.save_context({"input" : "Hi"}, {"output" : "Whats up Ramtin"})
print(memory.buffer)