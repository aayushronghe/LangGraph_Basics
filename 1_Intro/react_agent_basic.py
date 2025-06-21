from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.agents import initialize_agent,tool
from langchain_community.tools import TavilySearchResults

import os
from datetime import datetime

@tool
def get_system_time(format: str="%Y-%m-%d %H:%M:%S"):
    """Returns the current date and time in the specified format"""

    current_time = datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time

env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..",".env")
load_dotenv(env_path)

llm = ChatOpenAI(api_key = os.getenv("OPEN_AI_API_KEY"))

search_tool = TavilySearchResults(search_depth = "basic")

tools = [search_tool,get_system_time]

agent = initialize_agent(tools=tools,llm=llm,agent="zero-shot-react-description",verbose = True)

agent.invoke("Who won IPL 2025 and how many days ago was it from today??")