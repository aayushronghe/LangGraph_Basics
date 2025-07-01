from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import create_react_agent,tool
from langchain_community.tools import TavilySearchResults

import os
from datetime import datetime


@tool
def get_system_time(format: str="%Y-%m-%d %H:%M:%S") -> str:
    """Returns the current date and time in the specified format"""

    current_time = datetime.now()
    formatted_time = current_time.strftime(format)
    return str(formatted_time)

env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..",".env")
load_dotenv(env_path)

llm = ChatOpenAI(api_key = os.getenv("OPEN_AI_API_KEY"))

search_tool = TavilySearchResults()

tools = [search_tool,get_system_time]

react_prompt = hub.pull("hwchase17/react")

react_agent_runnable = create_react_agent(
    llm = llm,
    tools = tools,
    prompt = react_prompt
    )