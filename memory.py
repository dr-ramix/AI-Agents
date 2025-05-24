import os
import openai
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
# For memory management
from langchain.memory import ConversationBufferMemory
#For limiting the memory to a certain number of messages
from langchain.memory import ConversationBufferWindowMemory
# For token-based memory management reducing the number of tokens in the memory
from langchain.memory import ConversationTokenBufferMemory
# For summarizing the conversation history so we can reduce the number of tokens
from langchain.memory import ConversationSummaryBufferMemory

_ = load_dotenv(find_dotenv()) # read local .env file
api_key = os.environ['OPENAI_API_KEY']

# Create OpenAI client
client = OpenAI(api_key=api_key)

llm_model = "gpt-3.5-turbo"

llm = ChatOpenAI(temperature=0.0, model=llm_model)
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=False
)

conversation.predict(input="Hello, how are you?") # this will be the first message
conversation.predict(input="What is your name?") # this will be the second message
conversation.predict(input="What is 1 + 1") # this will be the third message
#print(memory.buffer)


# The memory will contain the conversation history
memory.load_memory_variables({}) # this will load the memory variables
#print(memory.load_memory_variables({}))

# How add new context to the memory 
memory = ConversationBufferMemory()
#memory.save_context({"input" : "Hi"}, {"output" : "Whats up Ramtin"})
#print(memory.buffer)


memory.save_context({"input": "Not much, just hanging"}, 
                    {"output": "Cool"})

#print(memory.load_memory_variables({}))

llm = ChatOpenAI(temperature=0.0, model=llm_model)
memory = ConversationBufferWindowMemory(k=1)
conversation = ConversationChain(
    llm=llm, 
    memory = memory,
    verbose=False
)

conversation.predict(input="Hi, my name is Ramtin")
conversation.predict(input="What is 1 + 1?") 
conversation.predict(input="What is my name?")
print(conversation.predict(input="What is my name?"))
print(memory.load_memory_variables({})) # this will load the memory variables


################ Using ConversationTokenBufferMemory to limit the number of tokens in the memory
memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=50)
memory.save_context({"input": "AI is what?!"},
                    {"output": "Amazing!"})
memory.save_context({"input": "Backpropagation is what?"},
                    {"output": "Beautiful!"})
memory.save_context({"input": "Chatbots are what?"}, 
                    {"output": "Charming!"})


## Summarizing the conversation history using ConversationSummaryBufferMemory
# create a long string
schedule = "There is a meeting at 8am with your product team. \
You will need your powerpoint presentation prepared. \
9am-12pm have time to work on your LangChain \
project which will go quickly because Langchain is such a powerful tool. \
At Noon, lunch at the italian resturant with a customer who is driving \
from over an hour away to meet you to understand the latest in AI. \
Be sure to bring your laptop to show the latest LLM demo."

memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=4)
memory.save_context({"input": "Hello"}, {"output": "What's up"})
memory.save_context({"input": "Not much, just hanging"},
                    {"output": "Cool"})
memory.save_context({"input": "What is on the schedule today?"}, 
                    {"output": f"{schedule}"})

print(memory.load_memory_variables({})) # this will load the memory variables