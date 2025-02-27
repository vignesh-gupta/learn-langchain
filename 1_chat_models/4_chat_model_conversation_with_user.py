from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from langchain_ollama import ChatOllama
from dotenv import load_dotenv

load_dotenv()

llm = ChatOllama(model="llama3.2")


chat_history = []

system_message = SystemMessage(content="You are an expert in Coding and Programming with 10+ years of experience in different languages and technologies. Give only one answer no explanation.")
chat_history.append(system_message)

while True:
  query = input("You: ")
  if query.lower() == "exit":
    break
  chat_history.append(HumanMessage(content=query))

  # Invoke the model with the chat history
  result = llm.invoke(chat_history)
  response = result.content
  chat_history.append(AIMessage(content=response))
  print("Bot:", response)


print("---- Messages ----")
print(chat_history)