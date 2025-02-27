from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableParallel

llm = ChatOllama(model="llama3.2")

summary_template = ChatPromptTemplate.from_messages([
    ("system", "You are a movie critic."),
    ("human", "Provide me a brief summary of the movie : {movie_name}")
])

# Define prompt template for analysis individually


def analyze_plot(plot):
    print("Analyzing plot...")
    plot_template = ChatPromptTemplate.from_messages([
        ("system", "You are a movie critic."),
        ("human",
            "Analyze plot is: {plot}. What are its strengths and weaknesses?")
    ])
    return plot_template.format_prompt(plot=plot)


def analyze_characters(characters):
    print("Analyzing characters...")
    character_template = ChatPromptTemplate.from_messages([
        ("system", "You are a movie critic."),
        ("human",
            "Analyze character is: {characters}. What are its strengths and weaknesses?")
    ])
    return character_template.format_prompt(characters=characters)


# Simplify branches with LCEL (LangChain Expression Language)
plot_branch_chain = (
    RunnableLambda(lambda x: analyze_plot(x)) | llm | StrOutputParser()
)
character_branch_chain = (
    RunnableLambda(lambda x: analyze_characters(x)) | llm | StrOutputParser()
)


# Combine the verdicts
def combine_verdicts(plot_output, characters_output):
    return f"Plot analysis: {plot_output}\n\nCharacter analysis: {characters_output}"


# Create the combined chain using LangChain Expression Language (LCEL)
chain = (
    summary_template
    | llm
    | StrOutputParser()
    | RunnableParallel(branches={"plot": plot_branch_chain, "characters": character_branch_chain})
    | RunnableLambda(lambda x: combine_verdicts(x["branches"]["plot"], x["branches"]["characters"])))


# Run the chain
print("Running the chain...")
result = chain.invoke({"movie_name": "Inception"})

print(result)
