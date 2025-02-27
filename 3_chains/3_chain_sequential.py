from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda

llm = ChatOllama(model="llama3.2")

# Main prompt template
fact_prompt_template=  ChatPromptTemplate.from_messages([
    ("system", "You are a facts expert who tells facts about {topic}."),
    ("human", "Tell me {fact_count} facts."),
])

#secondary prompt template
translate_prompt_template=  ChatPromptTemplate.from_messages([
    ("system", "You are a translator who translates provided text to {language}."),
    ("human", "The text is: {text}")
])

count_words = RunnableLambda(lambda x: f"Word count: {len(x.split())}\n{x}")
prepare_for_translation = RunnableLambda(lambda output: {"text": output, "language": "hindi"} )



# Create a chain 
# chain = fact_prompt_template | llm 
chain = fact_prompt_template | llm  | StrOutputParser() | prepare_for_translation | translate_prompt_template | llm | StrOutputParser() 

# Invoke the chain
result = chain.invoke({"topic": "cat", "fact_count": 1})

# Print the result
print(result)