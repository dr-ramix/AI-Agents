import os
import openai
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

_ = load_dotenv(find_dotenv()) # read local .env file
api_key = os.environ['OPENAI_API_KEY']

# Create OpenAI client
client = OpenAI(api_key=api_key)

llm_model = "gpt-3.5-turbo"


applicant_info = """
My name is Daniel Reyes, and I’m a passionate software engineer with a strong foundation in computer science and over four years of professional experience in backend and full-stack development. I take pride in writing clean, scalable code and solving real-world problems through technology.

I hold a Bachelor's degree in Computer Science from the University of Toronto, where I graduated with honors in 2019. During my time there, I focused on systems programming and AI, which sparked my interest in building intelligent and efficient applications. Since then, I’ve been working primarily with Python, Java, and JavaScript, and I have hands-on experience with frameworks like Django, Node.js, and React.

Currently, I work at CoreLink Technologies as a Software Engineer, where I lead the development of REST APIs and microservices for enterprise applications. I’ve also worked on DevOps automation using Docker and Kubernetes, and I’m well-versed in deploying applications on cloud platforms like AWS and GCP.

My technical toolkit includes a strong understanding of databases (PostgreSQL, MongoDB), CI/CD pipelines, version control with Git, and Agile methodologies. Beyond the technical side, I’m a collaborative team member and enjoy mentoring junior developers and participating in code reviews and tech discussions.

Outside of work, I’m an avid problem-solver—I frequently participate in hackathons and coding competitions. I also love staying active, whether it’s hiking, playing tennis, or practicing yoga. Music is another big part of my life; I’ve been playing the guitar for over 10 years and occasionally perform at local events.

I’m constantly learning and looking for opportunities to grow both personally and professionally. Whether it’s mastering a new programming language or exploring UX design, I enjoy diving into challenges and expanding my skill set.

"""


prompt_template = """\
For the following person, extract the following information:\
name: the name of this person\
skills: Which skills does he have\
experience: in Json (title: title of the job, company: company name, description: description of the job)\
education: in Json (title: title of the major, uni: uni name, degree: bachelor, master ...)\
interests: what are this persons interests\
hobbies: which hobby did he mentioned\
Format the output as a JSON object with following keys:\
name, skills, experience, education, interests, hobbies.\
If something is not mentioned, leave it as empty string like "".\
applicant_info: {applicant_info}\
"""
prompt_template = ChatPromptTemplate.from_template(prompt_template)
formatted_prompt = prompt_template.format(applicant_info=applicant_info)
print(formatted_prompt)


messages = prompt_template.format_messages(applicant_info=applicant_info)
chat = ChatOpenAI(temperature=0.0, model=llm_model)
response = chat(messages)
print(response.content)








name_schema = ResponseSchema(name="name",
                             description="Wthe name of this person"
                            )
skills_schema = ResponseSchema(name="skills",
                                      description="Which skills does he have"
                                    )
experience_schema = ResponseSchema(
    name="experience",
    description=(
        "A list of job objects with structure: "
        "[{title: title of the job, company: company name, description: description of the role, years: number of years}]"
    )
)

education_schema = ResponseSchema(
    name="education",
    description=(
        "A list of education records with structure: "
        "[{title: major or field of study, uni: university name, degree: bachelor/master/PhD, year: graduation year}]"
    )
)

interests_schema = ResponseSchema(name="interests",
                                    description="what are this persons interests"
                                  )
hobbies_schema = ResponseSchema(name="hobbies",
                                    description="which hobby did he mentioned"
                                )



response_schemas = [name_schema, skills_schema, experience_schema, education_schema, interests_schema, hobbies_schema]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()
print(format_instructions)



template2 = """\
For the following text, extract the following information:

For the following person, extract the following information:\
name: the name of this person\
skills: Which skills does he have\
experience: in Json (title: title of the job, company: company name, description: description of the job)\
education: in Json (title: title of the major, uni: uni name, degree: bachelor, master ...)\
interests: what are this persons interests\
hobbies: which hobby did he mentioned\
Format the output as a JSON object with following keys:\
name, skills, experience, education, interests, hobbies.\
If something is not mentioned, leave it as empty string like "".\
applicant_info: {applicant_info}\

{format_instructions}
"""

prompt = ChatPromptTemplate.from_template(template=template2)

messages = prompt.format_messages(applicant_info=applicant_info,
                                format_instructions=format_instructions)


response = chat(messages)
output_dict = output_parser.parse(response.content)
print(output_dict.get("education"))

