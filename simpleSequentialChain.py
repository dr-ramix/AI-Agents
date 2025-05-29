import os
import openai
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain


_ = load_dotenv(find_dotenv())  # read local .env file
api_key = os.environ['OPENAI_API_KEY']

client = OpenAI(api_key=api_key)


llm_model = "gpt-3.5-turbo"

llm = ChatOpenAI(temperature=0.9, model=llm_model)

first_prompt = ChatPromptTemplate.from_template(
    "Translate the following job description into German: {job_description}"
)

chain1 = LLMChain(llm=llm, prompt=first_prompt, output_key="german_translated_job_description")

second_prompt = ChatPromptTemplate.from_template(
    "Summarize the following job_description: {german_translated_job_description}"
)

chain2 = LLMChain(llm=llm, prompt=second_prompt, output_key="summary")

third_prompt = ChatPromptTemplate.from_template(
    "what language is the following job description written in: {job_description}"
)

chain3 = LLMChain(llm=llm, prompt=third_prompt, output_key="language")

fourth_prompt = ChatPromptTemplate.from_template(
"""Write a cover letter for the following summary in the specified language:

Job Description: {job_description}

Summary: {summary}

Language: {language}"""
)

chain4 = LLMChain(llm=llm, prompt=fourth_prompt, output_key="cover_letter")

# Create a sequential chain that runs the first three chains in order
overall_chain = SequentialChain(
    chains=[chain1, chain2, chain3, chain4],
    input_variables=["job_description"],
    output_variables=["german_translated_job_description", "summary", "language", "cover_letter"],
    verbose=True
)

job_description = """
Job Title: Technology Specialist (IT Support & Systems Administration)
Location: [Insert Location or "Remote"]
Job Type: Full-Time
Department: IT / Technical Services
Reports To: IT Manager / CTO

Job Summary:
We are seeking a detail-oriented and proactive Technology Specialist to join our growing IT team. The ideal candidate will be responsible for maintaining and improving our IT infrastructure, providing end-user support, and ensuring optimal system performance. This role involves troubleshooting hardware/software issues, managing networks, and supporting cloud-based applications and security protocols.

Key Responsibilities:
Provide technical support to staff via email, phone, or in-person.

Maintain, upgrade, and troubleshoot workstations, servers, and networking equipment.

Manage user accounts and permissions across multiple systems (e.g., Active Directory, Office 365, Google Workspace).

Install and configure software and hardware.

Monitor system performance and maintain data integrity and security.

Assist with IT asset inventory and procurement.

Support cloud systems and tools (e.g., AWS, Azure, SaaS platforms).

Document technical issues and solutions in support knowledge base.

Assist in the development and implementation of IT policies and procedures.

Collaborate with other departments to support technology needs.

Qualifications:
Bachelor’s degree in Information Technology, Computer Science, or related field (or equivalent experience).

2+ years of experience in IT support, systems administration, or similar role.

Strong understanding of operating systems (Windows, macOS, Linux).

Familiarity with networking protocols, routers, firewalls, and VPNs.

Experience with cloud platforms (AWS, Azure, or Google Cloud) is a plus.

Excellent problem-solving and communication skills.

Certifications such as CompTIA A+, Network+, or Microsoft certifications are a plus.

Benefits:
Competitive salary

Health, dental, and vision insurance

401(k) with company match

Generous PTO and paid holidays

Opportunities for training and career advancement

Flexible work environment
"""

print(overall_chain(job_description))