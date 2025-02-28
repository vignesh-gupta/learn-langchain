from langchain_ollama import ChatOllama
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor, tool
import datetime


@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """
    Get the current system time
    :param format: The format of the time
    :return: The current time in the specified format
    """
    return datetime.datetime.now().strftime(format)


llm = ChatOllama(model="llama3.2")
query = "What is current time? Provide the time in the format US format (MM-DD-YYYY HH:MM:SS)."

prompt_template = hub.pull("hwchase17/react")


tools = [get_system_time]

agent = create_react_agent(llm, tools, prompt_template)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke({"input": query})
