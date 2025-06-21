from langchain_core.prompts import MessagesPlaceholder,ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

import os

env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..",".env")
load_dotenv(env_path)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a techie influencer assisstant tasked with writing excellent twitter posts."
            "Generate the best twitter post possible for the user's request."
            "If the user provides critique, respond with a revised version of your previous attmepts."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user's tweet."
            "Always provide detailed recommendations, including requests for length, virality, style, etc.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

llm = ChatOpenAI(api_key=os.getenv("OPEN_AI_API_KEY"))

generation_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm