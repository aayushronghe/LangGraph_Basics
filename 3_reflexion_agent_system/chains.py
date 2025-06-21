from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import datetime
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.messages import HumanMessage

import os

from schema import AnswerQuestion

env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..",".env")
load_dotenv(env_path)

llm = ChatOpenAI(api_key=os.getenv("OPEN_AI_API_KEY"))

parser = PydanticToolsParser(tools=[AnswerQuestion])

actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert AI researcher,
            Current time: {time}
            
            1. {first_instruction}
            2. Reflect and critique your answer. Be severe to maximize improvement.
            3. After the reflection, ** list 1-3 search queries separately ** for research improvements. Do not include them inside the reflection.""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system","Answer the user's question above using the required fromat."),

    ]
).partial(
    time= lambda:datetime.datetime.now().isoformat(),
)

first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction = "Provide a detailed ~250 word answer"
)

first_responder_chain = first_responder_prompt_template | llm.bind_tools(tools=[AnswerQuestion],tool_choice='AnswerQuestion') | parser

response = first_responder_chain.invoke({
    "messages":[HumanMessage(content="Write me a blog post on how small business can leverage AI to grow")]
})

print(response)