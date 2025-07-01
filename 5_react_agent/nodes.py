from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode

from agent_reason_runnable import react_agent_runnable,tools
from react_state import AgentState

import os

env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..",".env")
load_dotenv(env_path)

def reason_node(state: AgentState):
    agent_outcome = react_agent_runnable.invoke(state)
    return {"agent_outcome": agent_outcome}

def act_node(state: AgentState):
    agent_action = state["agent_outcome"]
    
    tool_name = agent_action.tool
    tool_input = agent_action.tool_input

    tool_function = None
    for tool in tools:
        if tool.name == tool_name:
            tool_function = tool
            break
    
    if tool_function:
        if isinstance(tool_input,dict):
            output = tool_function.invoke(**tool_input)
        else:
            output = tool_function.invoke(tool_input)
    else:
        output = f"tool {tool_name} not found"
    
    return {"intermediate_steps":[(agent_action,str(output))]}