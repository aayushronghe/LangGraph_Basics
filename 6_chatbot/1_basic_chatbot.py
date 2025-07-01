from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langgraph.graph import END,StateGraph,add_messages
from langchain_core.messages import AIMessage,HumanMessage

from typing import Annotated,TypedDict
import os

env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..",".env")
load_dotenv(env_path)

llm = ChatGroq(model='llama-3.1-8b-instant')

class BasicChatState(TypedDict):
    messages: Annotated[list,add_messages]

def chatbot(state: BasicChatState):
    return {
        "messages": [llm.invoke(state['messages'])]
    }

graph = StateGraph(BasicChatState)

graph.add_node("chatbot",chatbot)
graph.set_entry_point("chatbot")
graph.add_edge("chatbot",END)

app = graph.compile()

while True:
    user_input = input("User: ")
    if user_input.lower() in ['exit','end','bye']:
        print("ChatBot: See you!!")
        break
    else:
        result = app.invoke({
            "messages":[HumanMessage(content=user_input)]
        })

        print("ChatBot: ",result['messages'][1].content)