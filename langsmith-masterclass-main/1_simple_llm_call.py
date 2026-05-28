from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Prompt
prompt = PromptTemplate.from_template("{question}")

# Updated Groq model
model = ChatGroq(
    model="llama-3.3-70b-versatile"
)

parser = StrOutputParser()

# Chain
chain = prompt | model | parser

# Run
result = chain.invoke({
    "question": "What is the capital of INDIA?"
})

print(result)