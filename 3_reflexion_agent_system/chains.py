from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import datetime
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.messages import HumanMessage

import os

from schema import AnswerQuestion,ReviseAnswer

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

revise_instructions = """
Revise your previous using the new information.
    - You should use the previous critique to add important information to your answer.
        - You must include numerical citations in your revised answer to ensure it can be verified.
        - Add a "Reference" section to the bottom of your answer ( which does not count towards the word limit ). In form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information form your answer and make sure it is not more than 250 words
"""

first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction = "Provide a detailed ~250 word answer"
)

first_responder_chain = first_responder_prompt_template | llm.bind_tools(tools=[AnswerQuestion],tool_choice='AnswerQuestion')

revisor_chain = actor_prompt_template.partial(
    first_instruction = revise_instructions
) | llm.bind_tools(tools=[ReviseAnswer],tool_choice='ReviseAnswer') 

response = first_responder_chain.invoke({
    "messages":[HumanMessage(content="Write me a blog post on how small business can leverage AI to grow")]
})
