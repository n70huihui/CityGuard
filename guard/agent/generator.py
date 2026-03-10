from langchain.agents.structured_output import ToolStrategy

from env_utils.llm_args import *

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from guard.common.model import FinalReport

generator = create_agent(
    model=ChatOpenAI(model=visual_model, base_url=base_url, api_key=api_key),
    tools=[],
    response_format=ToolStrategy(FinalReport)
)