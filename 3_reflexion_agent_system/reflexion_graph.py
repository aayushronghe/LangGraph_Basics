from typing import List
from langchain_core.messages import BaseMessage,ToolMessage
from langgraph.graph import MessageGraph,END

from execute_tools import execute_tools
from chains import revisor_chain,first_responder_chain

graph = MessageGraph()

graph.add_node("draft",first_responder_chain)
graph.add_node("execute_tools",execute_tools)
graph.add_node("revisor",revisor_chain)

graph.add_edge("draft","execute_tools")
graph.add_edge("execute_tools","revisor")

def event_loop(state: List[BaseMessage]) -> str:
    count_tool_visits = sum(isinstance(item,ToolMessage) for item in state)
    num_iterations = count_tool_visits
    if num_iterations > 2:
        return END
    return "execute_tools"

graph.add_conditional_edges("revisor",event_loop)
graph.set_entry_point("draft")

app = graph.compile()

response = app.invoke("Write about how small business can leverage AI to grow")

print(response[-1].tool_calls[0]['args']['answer'])