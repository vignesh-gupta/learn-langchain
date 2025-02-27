from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

llm = ChatOllama(model="llama3.2")

prompt_template=  ChatPromptTemplate.from_messages([
    ("system", "You are a facts expert who tells facts about {topic}."),
    ("human", "Tell me {fact_count} facts."),
])

# Create a chain 
# chain = prompt_template | llm 
chain = prompt_template | llm  | StrOutputParser()

# Invoke the chain
result = chain.invoke({"topic": "dogs", "fact_count": 3})

# Print the result
print(result)