import getpass
import os
os.environ["OPENAI_API_KEY"] = "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"
os.environ["TAVILY_API_KEY"] ="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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "ls__ae07f6c2e7844ccfa35d0814bbef82d4"
os.environ['SERPAPI_API_KEY'] = "b5dd0700a83fc3193772b29cb9bcf30ec03a0b85ae10d63296ea95e40f489630"
os.environ['SERPER_API_KEY'] = "d54d37471d851575ce40b0f3925d2c6ac5079549"
from web_searchagent import main
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
new_tools = load_tools(["google-serper"], llm=llm)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful assistant. Youre task is to use tools to resolve the python astronomical problem

            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{messages}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)
# llm_with_tools = llm.bind_tools(tools)
agent = create_tool_calling_agent(llm, new_tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=new_tools, verbose=True)

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
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

Use the tools at each step to first get the information from the browser and then use that to code the result
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
    """. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

   
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

code_total = []
code_summary = []
async def execute_step(state: PlanExecute):
    plan = state["plan"]
    plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
    task = plan[0]
    task_formatted = f"""For the following plan:
{plan_str}\n\nYou are tasked with executing step {1}, {task}."""
    response_content,response,code_point = await execute_python(task, state["input"], state["plan"], state["past_steps"])
    return {
                "past_steps": (task,response_content , response , "Assume that this response is correct and if the question has a graph if the code is correct the graph has been plotted",code_point , "This is the code executed in this step")
            }

from openai import OpenAI
def summarize_code():
    print("HERE IT BREAKS")
    if not code_total:  # Check if code_total is empty
        print("No code to summarize.")
        return
    
    # Access the last element safely
    latest_code_snippet = code_total[-1] if code_total else None
    if not latest_code_snippet:  # Check if the last element is None or empty
        print("The latest code snippet is empty.")
        return
    client = OpenAI()
    response = client.chat.completions.create(
    model="gpt-3.5-turbo-0125",
    messages=[
            {"role": "system", "content": "You are a helpful assistant designed to summarize python code in detail."},
            {"role": "user", "content": f"Can you summarize the python code for me {latest_code_snippet}."}
    ]
    )
    print("THIS IS THE code_summary till now")
    answer = (response.choices[0].message.content).strip().lower()
    code_summary.append(answer)
    print(code_summary)
    # return {"requires_python": answer == "yes"}
from codeinterpreterapi import CodeInterpreterSession, settings
session_id = None
session = CodeInterpreterSession()
session.start()
async def execute_python(task,input_val, plan, past_steps):
    """this step is to execute the python code """
    summarize_code()
    print("IM in executing python code")

    new_prompt = f"""You are an excellent python programmer and your task is to execute the python code based on the plan
    Do not add any superfluous steps. or functions that are not defined, if you call a function or variable ensure it is correctly defined and remember your overall code is dependent on the previous steps and will continue with more steps 
    Your code is constantly being written to a file as well and being executed so make sure not to write redundant code.
    A summary of the code will be provided below.
    Your code should be completely executable and nothing should be left as a placeholder.
    Your objective was this:
    {input_val}

    Your original plan was this:
    {' '.join(plan)}

    
    You have currently done the follow steps:
    {past_steps}


    Your current code is already written and does not need to be written again. A summary of the written code is provided below
    {code_summary}

    Based on this step program the current step {task}



    """
    print(new_prompt)
    settings.OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    settings.MODEL = "gpt-4-turbo-preview"

    response = session.generate_response(new_prompt)

    # print(response.code_log[0][0] ,"this is the code in that step")
    try:
        if response.code_log and isinstance(response.code_log[0], list):
            print(f"here in if one {response.code_log[0]}")
            code_to_execute = response.code_log[0][0]  # Expected to be the actual code snippet
            code_total.append(code_to_execute)
        elif response.code_log:
            print(f"here in if one {response.code_log}")
            code_to_execute = response.code_log  # A single list with code directly
            code_total.append(code_to_execute)
        else:
            print("No executable code found in response.")
            return response.content,response,None
        
        print(f"Executing code: {code_to_execute}")
        await execute_code(code_to_execute)
        return response.content,response,code_to_execute
    except IndexError as e:
        print(f"Error accessing code to execute: {e}")
        await execute_code(code_to_execute)
        return response.content,response, code_to_execute
    except Exception as e:
        print(f"An error occurred: {e}")
        await execute_code(code_to_execute)
        return response.content,response ,code_to_execute
import subprocess
import tempfile
import os

async def execute_code(code):
    print("IM IN CODE \n", code)
    if isinstance(code, list):
        code = '\n'.join([c[0] if isinstance(c, tuple) else c for c in code])
    temp_dir = r"testing"
    os.makedirs(temp_dir, exist_ok=True)
    
    # with tempfile.NamedTemporaryFile(delete=False, suffix='.py', mode='w+t', encoding='utf-8', dir=temp_dir) as temp_file:
    #     temp_file_name = temp_file.name
    #     print("Temporary file created:", temp_file_name)
    #     temp_file.write(code)
    #     temp_file.flush()
    
    # result = subprocess.run(["python", temp_file_name], capture_output=True, text=True)
    temp_file_path = os.path.join(temp_dir, 'temp_file5.py')
    with open(temp_file_path, 'a', encoding='utf-8') as temp_file:
        print("Writing code to:", temp_file_path)
        temp_file.write(code + '\n')
    
    result = subprocess.run(["python", temp_file_path], capture_output=True, text=True)
    if result.stderr:
        error_message = "Error: " + result.stderr
        return None, error_message
    else:
        return result.stdout, None


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
async def main():
    config = {"recursion_limit": 50}
    inputs = {"input": """
Prompt Question #3:
A Simple Protostellar Evolution Model

Consider the evolution of a protostar forming with a constant accretion rate $\dot{M}$. As gas falls freely onto the protostar, it radiates a luminosity $L_{\rm acc}$ at the accretion shock. After contraction, the resulting star is fully ionized, with all deuterium converted to hydrogen and in hydrostatic equilibrium. Key energies involved include hydrogen ionization potential, molecular hydrogen dissociation potential, and energy released from deuterium burning.

1. Total Energy Calculation: For a low-mass protostar described by an $n=3/2$ polytrope, compute the total energy, considering thermal, gravitational, and chemical energies.

2. Evolution Equation Derivation: Derive an equation for the star's radius evolution, assuming it follows the Hayashi track with a fixed effective temperature.

3. PYTHON: Numerical Integration and Plotting: Integrate the evolution equation numerically and plot radius and luminosity against mass for given accretion parameters. Start integration from specified initial conditions and stop at $M=1.0$ $\msun$.

4. Modifications for Massive Protostars: Adapt the model for massive protostars by considering a polytropic index $n=3$ and a luminosity function. 
Modify the evolution equation accordingly and integrate numerically up to $M=50$ $\msun$.
"""}
    async for event in app.astream(inputs, config=config):
        for k, v in event.items():
            if k != "__end__":
                print(v)
if __name__ == "__main__":
    asyncio.run(main())