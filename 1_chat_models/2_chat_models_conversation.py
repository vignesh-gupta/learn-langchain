from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from langchain_ollama import ChatOllama
from dotenv import load_dotenv

load_dotenv()

llm = ChatOllama(model="llama3.2")


messages = [
  SystemMessage("You are an expert in Coding and Programming with 10+ years of experience in different languages and technologies."),
  HumanMessage("What is the best programming language for beginners to get into web? Give only one answer no explanation."),
]
result = llm.invoke(messages)

print(result.content)