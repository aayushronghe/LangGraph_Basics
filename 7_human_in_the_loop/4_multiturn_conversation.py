from langgraph.graph import START,END,StateGraph,add_messages
from langgraph.types import Command,interrupt
from typing import TypedDict,Annotated,List
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import uuid

llm = ChatGroq(model = 'llama-3.1-8b-instant')

class State(TypedDict):
    linkedin_topics: str
    generated_post: Annotated[List[str],add_messages]
    human_feedback: Annotated[List[str],add_messages]

def model(state: State):
    """Here, we're using the LLM to generate a LinkedIn post with human feedback incorporated"""

    print('[model] Generating content')
    linkedin_topic = state['linkedin_topics']
    feedback = state['human_feedback'] if 'human_feedback' in state else ['No feedback yet']

    prompt = f"""

        LinkedIn Topic: {linkedin_topic}
        Human Feedback: {feedback[-1] if feedback else "No feedback yet"}

        Generate a structured and well-written LinkedIn post based on the given topic.

        Consider previous human feedback to refine the reponse. 
    """

    response = llm.invoke([
        SystemMessage(content='You are an expert LinkedIn content writer'),
        HumanMessage(content=prompt)
    ])

    genreated_response = response.content

    print(f'[model_node] Generated post:\n {genreated_response}')

    return{
        'generated_post':[AIMessage(content=genreated_response)],
        'human_feedback': feedback
    }

def human_node(state: State):
    """Human intervention node - loops back to model unless input is done"""

    print("\n [human_node] Awating human feedback...")

    generated_post = state['generated_post']

    user_feedback = interrupt(
        {
            'generated_post': generated_post,
            'message':"Provide feedback or type 'done' to finish "
        }
    )

    print(f"[human_node] Received human feedback: {user_feedback}")

    if user_feedback.lower()=='done':
        return Command(update={'human_feedback': state['human_feedback'] + ['Finalised']},goto='end_node')
    
    return Command(update={'human_feedback': state['human_feedback'] + [user_feedback]}, goto='model')

def end_node(state: State):
    """Final node"""

    print("\n[end_node]: Process Finished!!")
    print(f"\nFinal Generated Post: {state['generated_post'][-1].content}")
    print(f"\nFinal Human Feedback: {state['human_feedback'][-1].content}")
    return{
        'generated_post':state['generated_post'],
        'human_feedback':state['human_feedback'],
    }

graph = StateGraph(State)
graph.add_node('model',model)
graph.add_node('human_node',human_node)
graph.add_node('end_node',end_node)

graph.add_edge(START,'model')
graph.add_edge('model','human_node')

graph.set_finish_point('end_node')

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

thread_config = {
    'configurable':{
        'thread_id':uuid.uuid4()
    }
}

linkedin_topic = input("Enter your LinkedIn topic: ")
initial_state={
    'linkedin_topics': linkedin_topic,
    'generated_post': [],
    'human_feedback': []
}

for chunk in app.stream(initial_state,config=thread_config):
    for node_id,value in chunk.items():
        if(node_id == "__interrupt__"):
            while True:
                user_feedback = input("Provide feedback (or type done when finished): ")

                app.invoke(Command(resume=user_feedback),config=thread_config)

                if user_feedback.lower() == 'done':
                    break