from typing import TypedDict, Annotated
from langgraph.graph import StateGraph,END,add_messages
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv

import os

env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..",".env")
load_dotenv(env_path)

class State(TypedDict):
    messages: Annotated[list,add_messages]

llm = ChatGroq(model='llama-3.1-8b-instant')

GENERATE_POST = 'generate_post'
GET_REVIEW_DECISION = 'get_review_decision'
POST = 'post'
COLLECT_FEEDBACK = 'collect_feedback'

def generate_post(state: State):
    return {
        'messages':[llm.invoke(state['messages'])]
    }

def get_review_decision(state: State):
    post_content = state['messages'][-1].content

    print('\n Current LinkedIn Post:\n')
    print(post_content)
    print('\n')

    decision = input("Post to LinkedIn? (yes/no): ")

    if decision.lower() == "yes":
        return POST
    else:
        return COLLECT_FEEDBACK

def post(state: State):
    final_post = state['messages'][-1].content
    print("\n Final LinkedIn Post: \n")
    print(final_post)
    print('\n')

def collect_feedback(state: State):
    feedback = input("How can I improve the post?\n Answer: ")
    return{
        'messages': [HumanMessage(content=feedback)]
    }

graph = StateGraph(State)

graph.add_node(GENERATE_POST,generate_post)
graph.set_entry_point(GENERATE_POST)

graph.add_node(GET_REVIEW_DECISION,get_review_decision)
graph.add_node(COLLECT_FEEDBACK,collect_feedback)
graph.add_node(POST,post)

graph.add_conditional_edges(GENERATE_POST,get_review_decision)
graph.add_edge(POST,END)
graph.add_edge(COLLECT_FEEDBACK,GENERATE_POST)

app = graph.compile()

response = app.invoke({
    'messages': HumanMessage(content='Give me a LinkedIn post on Misuse of AI in modern world')
})

print(response['messages'][-1].content)