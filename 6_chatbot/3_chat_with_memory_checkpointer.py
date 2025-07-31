from langgraph.graph import StateGraph,END,add_messages
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

from typing import TypedDict, Annotated
import os

env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..",".env")
load_dotenv(env_path)

llm = ChatGroq(model='llama-3.1-8b-instant')

memory = MemorySaver()

class BasicChatState(TypedDict):
    messages: Annotated[list,add_messages]

def chatbot(state: BasicChatState):
    return {
        'messages':[llm.invoke(state['messages'])]
    }

graph = StateGraph(BasicChatState)

graph.add_node("chatbot",chatbot)

graph.set_entry_point('chatbot')

graph.add_edge('chatbot',END)

app = graph.compile(checkpointer=memory)

config = {"configurable":{
    "thread_id":1
}}

while True:
    user_input = input("User: ")
    if(user_input.lower() in ['exit','bye','thanks','see ya']):
        print('Chatbot: See ya!!')
        break
    else:
        result = app.invoke({
            'messages': HumanMessage(content=user_input)
        },config=config)
        print('Chatbot: ',result['messages'][-1].content)