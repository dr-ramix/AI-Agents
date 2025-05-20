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
# def get_completion(prompt, model=llm_model):
#     messages = [{"role": "user", "content": prompt}]
#     response = client.chat.completions.create(
#         model=model,
#         messages=messages,
#         temperature=0
#     )
#     return response.choices[0].message.content

# print(get_completion("Write me cover letter with 1000 words for a Django developer posion at IBM, be creative"))

# customer_skills = """
# Python, CSS, HTML, DOCKER, Node JS, React JS, Machine Learning, JavaScript, CI/CD, SQL, MongoDB
# """

# style = """American English \
# respectful tone, creative like a story, strong, confident, based on the job description, base on the facts\
# """

# prompt = f"""Write me cover letter with 500 words for a Django developer posion at IBM, be creative\
# into a style that is {style}.
# with skills: ```{customer_skills}```
# """

# response = get_completion(prompt)

# print(response)


# text by an LLM, use temperature = 0.0
chat = ChatOpenAI(temperature=0.0, model=llm_model)

name = "John Doe"
age = 30
location = "Seattle, WA"
about = "Aisha Khan is a results-driven software engineer with over 3 years of experience developing scalable web applications and backend services. She has a strong foundation in computer science principles and a proven ability to work across the full software development lifecycle. Aisha thrives in collaborative environments and is passionate about writing clean, efficient code and staying up to date with the latest in cloud and container technologies."
skills = "Python, Java, JavaScript (ES6+), React, Django, Spring Boot, AWS (EC2, S3, Lambda), Docker, Kubernetes, GitHub Actions, PostgreSQL, MongoDB, Git, JIRA, VS Code, Postman, Agile/Scrum, CI/CD, TDD"
experience = "Software Engineer – Orion Systems Inc., Seattle, WA, Jan 2022 – Present, Developed and maintained microservices using Django and Spring Boot, Built responsive frontend components with React and integrated REST APIs, Led migration of legacy systems to AWS cloud improving system uptime by 30%, Containerized applications using Docker and deployed with Kubernetes, Worked in Agile sprints actively participating in daily stand-ups and retrospectives, Junior Software Developer – Hexabyte Innovations, Remote, Aug 2020 – Dec 2021, Contributed to a client-facing platform built in Python and JavaScrip"
education  = "B.S. in Computer Science, University of Washington (2016–2020)"


company_name = "NovaTech Solutions"
job_title = " Software Engineer"
job_description = """
    NovaTech Solutions is seeking a highly motivated Software Engineer to join our growing development team. In this role, you will be responsible for designing, developing, testing, and maintaining scalable software applications that power critical business operations and next-gen customer experiences.

Responsibilities:

Collaborate with cross-functional teams to define, design, and ship new features.

Write clean, maintainable, and efficient code in languages such as Python, Java, or JavaScript.

Build and maintain APIs and microservices that support our frontend and mobile applications.

Participate in code reviews and contribute to continuous improvement initiatives.

Debug and resolve technical issues across the full stack.

Stay current with emerging technologies and propose innovative solutions.

Requirements:

Bachelor's degree in Computer Science, Software Engineering, or a related field.

2+ years of experience in software development.

Proficiency in one or more programming languages (e.g., Python, Java, C#, JavaScript).

Familiarity with modern frameworks (e.g., React, Django, Spring Boot).

Experience with version control (Git) and CI/CD pipelines.

Strong problem-solving and communication skills.

Preferred Qualifications:

Experience with cloud platforms (AWS, Azure, or GCP).

Knowledge of containerization tools like Docker and Kubernetes.

Familiarity with Agile/Scrum methodologies.

"""


template_string = """Write a cover 500 words letter for this person\
Name: {name}\
Age: {age}\
Location: {location}\
About: {about}\
Skills: {skills}\
Experience: {experience}\
Education: {education}\
for this job:\
Company Name: {company_name}\
Job Title: {job_title}\
Job Description: {job_description}\
into a style that is {style}. \
"""

prompt_template = ChatPromptTemplate.from_template(template_string)

print(prompt_template.messages[0].prompt)
print(prompt_template.messages[0].prompt.input_variables)


cover_letter_info = prompt_template.format_messages(
    name=name,
    age=age,
    location=location,
    about=about,
    skills=skills,
    experience=experience,
    education=education,
    company_name=company_name,
    job_title=job_title,
    job_description=job_description,
    style="American English, respectful tone, creative like a story, strong, confident, based on the job description, base on the facts",
) 

print(type(cover_letter_info))
print(type(cover_letter_info[0]))

print(cover_letter_info[0])

cover_letter = chat(cover_letter_info)

print(cover_letter.content)



