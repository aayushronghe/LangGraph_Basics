from langgraph.graph import StateGraph, END, add_messages
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage,AIMessage
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode

from typing import Annotated,TypedDict
import os

env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..",".env")
load_dotenv(env_path)

search_tool = TavilySearchResults(max_results = 2)
tools = [search_tool]

llm = ChatGroq(model = 'llama-3.1-8b-instant')

llm_with_tools = llm.bind_tools(tools=tools)

class BasicChatBot(TypedDict):
    messages: Annotated[list,add_messages]

def chatbot(state: BasicChatBot):
    return{
        'messages':[llm_with_tools.invoke(state['messages'])]
    }

def tool_router(state: BasicChatBot):
    last_message = state['messages'][-1]

    if(hasattr(last_message,'tool_calls') and len(last_message.tool_calls)>0):
        return 'tool_node'
    else:
        return END
    
tool_node = ToolNode(tools)

graph = StateGraph(BasicChatBot)

graph.add_node('chatbot',chatbot)
graph.set_entry_point('chatbot')
graph.add_node('tool_node',tool_node)
graph.add_conditional_edges('chatbot',tool_router)
graph.add_edge('tool_node','chatbot')

app = graph.compile()

while True:
    user_input = input("User: ")
    if(user_input.lower() in ['bye','exit','end']):
        print("Chatbot: See ya!!!")
        break
    else:
        result = app.invoke({
            'messages':[HumanMessage(content=user_input)]
        })

        print("Chatbot: ",result['messages'][-1].content)
