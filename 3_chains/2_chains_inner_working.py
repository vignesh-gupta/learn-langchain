from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda,RunnableSequence

llm = ChatOllama(model="llama3.2")

prompt_template=  ChatPromptTemplate.from_messages([
    ("system", "You are a facts expert who tells facts about {topic}."),
    ("human", "Tell me {fact_count} facts."),
])

format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
invoke_llm = RunnableLambda(lambda x: llm.invoke(x.to_messages()))
parse_output = RunnableLambda(lambda x: x.content)

chain = RunnableSequence(first=format_prompt, middle=[invoke_llm], last=parse_output)

# Invoke the chain
result = chain.invoke({"topic": "dogs", "fact_count": 3})

# Print the result
print(result)