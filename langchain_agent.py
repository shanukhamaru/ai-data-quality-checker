import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load environment variables here!
load_dotenv()

# Now get the key AFTER loading .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize LLM
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)

prompt = PromptTemplate(
    input_variables=["issues"],
    template="""
You are a data scientist. Given these data quality issues:

{issues}

Suggest smart, practical data cleaning strategies for each issue. Be concise and clear.
"""
)

chain = LLMChain(llm=llm, prompt=prompt)

def get_cleaning_suggestions(issues_dict: dict):
    issues_str = "\n".join([f"{k}: {v}" for k, v in issues_dict.items()])
    response = chain.run(issues=issues_str)
    return response
