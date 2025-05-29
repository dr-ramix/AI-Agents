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

from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain,RouterOutputParser
from langchain.prompts import PromptTemplate



_ = load_dotenv(find_dotenv()) # read local .env file
api_key = os.environ['OPENAI_API_KEY']

# Create OpenAI client
client = OpenAI(api_key=api_key)

# Define the LLM model
llm_model = "gpt-3.5-turbo"

#create a model for the LLM
llm = ChatOpenAI(temperature=0, model=llm_model)


#Templates:
physics_interviewer_template = """You are a senior physicist conducting a job interview for a research or engineering position. \
You have deep expertise in applied physics, problem-solving, and scientific reasoning. \
You assess candidates by asking technical questions that test their understanding of physical principles, \
their ability to design and analyze experiments, and their skill in applying theory to practical problems.

Conduct an interview. Ask a technical job interview question about:
{input}"""

math_interviewer_template = """You are a quantitative analyst and senior mathematician conducting a job interview for a quantitative or data-focused role. \
You evaluate candidates by asking challenging questions that test mathematical modeling, logic, and problem-solving skills. \
You prefer questions that require mental math, algorithmic thinking, and rigorous reasoning.

Conduct an interview. Ask a technical job interview question about:
{input}"""



data_science_interviewer_template = """You are a lead data scientist conducting a technical job interview for a data science or machine learning role. \
You evaluate candidates based on their understanding of statistics, modeling, data analysis, and business relevance. \
You ask practical questions that require interpreting data, building models, and communicating insights clearly.

Conduct an interview. Ask a technical job interview question about:
{input}"""


software_engineer_interviewer_template = """You are a principal software engineer conducting a job interview for a backend or full-stack developer role. \
You assess candidates on software design, coding, debugging, scalability, and team collaboration. \
You ask technical questions that simulate real-world scenarios involving architecture, performance, and maintainability.

Conduct an interview. Ask a technical job interview question about:
{input}"""


prompt_infos = [
    {
        "name": "physics_interviewer",
        "template": physics_interviewer_template,
        "description": "Good for physics-related job interviews, focusing on applied physics and problem-solving.",
    },
    {
        "name": "math_interviewer",
        "template": math_interviewer_template,
        "description": "Ideal for quantitative roles, emphasizing mathematical modeling and algorithmic thinking.",
    },
    {
        "name": "data_science_interviewer",
        "template": data_science_interviewer_template,
        "description": "Best for data science and machine learning positions, focusing on statistics and data analysis.",
    },
    {
        "name": "software_engineer_interviewer",
        "template": software_engineer_interviewer_template,
        "description": "Suitable for software engineering roles, assessing coding, design, and system architecture skills.",
    }
]


destination_chains = {}
for prompt_info in prompt_infos:
    name = prompt_info["name"]
    prompt_template = prompt_info["template"]
    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    destination_chains[name] = chain

destinations = [f"{p["name"]}: {p["description"]}" for p in prompt_infos]
destinations_str = "\n".join(destinations)

default_prompt = ChatPromptTemplate.from_template("{input}")
default_chain = LLMChain(llm=llm, prompt=default_prompt)




MULTI_PROMPT_ROUTER_TEMPLATE = """Given a raw text input with a job title to a \
language model select the model prompt best suited for the input. \
You will be given the names of the available prompts and a \
description of what the prompt is best suited for. \
You may also revise the original input if you think that revising\
it will ultimately lead to a better response from the language model.

<< FORMATTING >>
Return a markdown code snippet with a JSON object formatted to look like:
```json
{{{{
    "destination": string \ "DEFAULT" or name of the prompt to use in {destinations},
    "description": string \ "DEFAULT" or description of the prompt to use in {destinations},
    "next_inputs": string \ a potentially Response of the destination to the the original input
}}}}
```

REMEMBER: The value of “destination” MUST match one of \
the candidate prompts listed below.\
If “destination” does not fit any of the specified prompts, set it to “DEFAULT.”
REMEMBER: "next_inputs" must be interview questions \


<< CANDIDATE PROMPTS >>
{destinations}

<< INPUT >>
{{input}}

<< OUTPUT (remember to include the ```json)>>"""


router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
    destinations=destinations_str
)
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
)

router_chain = LLMRouterChain.from_llm(llm, router_prompt)

chain = MultiPromptChain(router_chain=router_chain, 
                         destination_chains=destination_chains, 
                         default_chain=default_chain, verbose=True
                        )

chain.run("This job applicant is applying for a data science position with python and R. Ask 10 questions")

