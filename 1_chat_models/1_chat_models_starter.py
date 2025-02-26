from langchain_ollama import ChatOllama
from dotenv import load_dotenv

load_dotenv()

llm = ChatOllama(model="llama3.2")

result = llm.invoke("What is square root of 49? give me the answer only no bs")

print(result.content)