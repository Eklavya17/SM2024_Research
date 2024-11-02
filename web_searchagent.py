import getpass
import os
os.environ["OPENAI_API_KEY"] = "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"
os.environ["TAVILY_API_KEY"] ="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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "ls__ae07f6c2e7844ccfa35d0814bbef82d4"
os.environ['SERPAPI_API_KEY'] = "b5dd0700a83fc3193772b29cb9bcf30ec03a0b85ae10d63296ea95e40f489630"
os.environ['SERPER_API_KEY'] = "d54d37471d851575ce40b0f3925d2c6ac5079549"
def _set_if_undefined(var: str) -> None:
    if os.environ.get(var):
        return
    os.environ[var] = getpass.getpass(var)


# Optional: Configure tracing to visualize and debug the agent
_set_if_undefined("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "LATS"

_set_if_undefined("OPENAI_API_KEY")
_set_if_undefined("TAVILY_API_KEY")
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain_openai import OpenAI
# llm = OpenAI(temperature=0)
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from langchain import hub
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import tool_executor

from langchain import OpenAI, SerpAPIWrapper
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
# search = SerpAPIWrapper()
# tools = [
#     Tool(
#         name="Intermediate Answer",
#         func=search.run,
#         description='google search'
#     )
# ]

# self_ask_with_search = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True,handle_parsing_errors=True)
# self_ask_with_search.run("what is the date today")
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
# new_tools = load_tools(["google-serper"], llm=llm)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful assistant. Make sure to use the internet search custom tool for getting ANY information.
            Given a user's question in natural language, suggest a list of keyword-based queries optimized for search engines. 

            Examples:
            1. When searching Google:
            - Input: "What CV courses are taught at UIUC?"
            - Output: ["UIUC computer vision curriculum", "University of Illinois Urbana-Champaign CV courses", "UIUC course explorer computer vision"]

            2. When searching Amazon:
            - Input: "Cooling solutions for outdoor activities?"
            - Output: ["portable cooling devices outdoor", "outdoor cooling gear"]

            Now, generate keyword-based queries for the following user questions and search those queries

            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{messages}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)
# llm_with_tools = llm.bind_tools(tools)
# agent = create_tool_calling_agent(llm, new_tools, prompt)
# agent_executor = AgentExecutor(agent=agent, tools=new_tools, verbose=True)

# tools
from typing import List, Tuple, Annotated, TypedDict
import operator


class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str
from langchain_core.pydantic_v1 import BaseModel, Field


class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )
from langchain_core.prompts import ChatPromptTemplate

planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """For the given objective, come up with a simple step by step plan
            . \
Your task is to breakdown the prompt into better google search queries that are optimized based on keyword breakdown
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.
All steps will involve google search

Breakdown the query into steps with optimized keyword searching that will give the best results instead of just searching the query as a whole.

Suggest a list of keyword-based queries optimized for search engines. 

Examples:
1. When searching Google:
   - Input: "What CV courses are taught at UIUC?"
   - Output: ["UIUC computer vision curriculum", "University of Illinois Urbana-Champaign CV courses", "UIUC course explorer computer vision"]

2. When searching Amazon:
   - Input: "Cooling solutions for outdoor activities?"
   - Output: ["portable cooling devices outdoor", "outdoor cooling gear"]
""",
        ),
        ("placeholder", "{messages}"),
    ]
)
planner = planner_prompt | ChatOpenAI(
    model="gpt-4o", temperature=0
).with_structured_output(Plan)
from typing import Union


class Response(BaseModel):
    """Response to user."""

    response: str


class Act(BaseModel):
    """Action to perform."""

    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )


replanner_prompt = ChatPromptTemplate.from_template(
    """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.
All steps will involve google search to be resolved.

Breakdown the query into steps with optimized keyword searching that will give the best results instead of just searching the query as a whole.

Suggest a list of keyword-based queries optimized for search engines. 

Examples:
1. When searching Google:
   - Input: "What CV courses are taught at UIUC?"
   - Output: ["UIUC computer vision curriculum", "University of Illinois Urbana-Champaign CV courses", "UIUC course explorer computer vision"]

2. When searching Amazon:
   - Input: "Cooling solutions for outdoor activities?"
   - Output: ["portable cooling devices outdoor", "outdoor cooling gear"]
   
Your objective was this:
{input}

Your original plan was this:
{plan}

You have currently done the follow steps:
{past_steps}

Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan."""
)


replanner = replanner_prompt | ChatOpenAI(
    model="gpt-4o", temperature=0
).with_structured_output(Act)
from typing import Literal


async def execute_step(state: PlanExecute):
    plan = state["plan"]
    plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
    task = plan[0]
    task_formatted = f"""For the following plan:
{plan_str}\n\nYou are tasked with executing step {1}, {task}."""
    agent_response = await agent_executor.ainvoke(
        {"messages": [("user", task_formatted)]}
    )
    return {
        "past_steps": (task, agent_response["output"]),
    }


async def plan_step(state: PlanExecute):
    plan = await planner.ainvoke({"messages": [("user", state["input"])]})
    return {"plan": plan.steps}


async def replan_step(state: PlanExecute):
    output = await replanner.ainvoke(state)
    if isinstance(output.action, Response):
        return {"response": output.action.response}
    else:
        return {"plan": output.action.steps}


def should_end(state: PlanExecute) -> Literal["agent", "__end__"]:
    if "response" in state and state["response"]:
        return "__end__"
    else:
        return "agent"
from langgraph.graph import StateGraph

workflow = StateGraph(PlanExecute)

workflow.add_node("planner", plan_step)

workflow.add_node("agent", execute_step)

workflow.add_node("replan", replan_step)

workflow.set_entry_point("planner")

workflow.add_edge("planner", "agent")

workflow.add_edge("agent", "replan")

workflow.add_conditional_edges(
    "replan",
    should_end,
)

app = workflow.compile()
import asyncio
async def main(input_s: str):
    config = {"recursion_limit": 50}
    inputs = {"input": input_s}
    async for event in app.astream(inputs, config=config):
        for k, v in event.items():
            if k != "__end__":
                print(v)
if __name__ == "__main__":
    asyncio.run(main())