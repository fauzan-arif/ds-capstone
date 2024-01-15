from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())

from datetime import datetime, timedelta
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain.agents.tools import Tool
from langchain.agents import AgentExecutor, create_openai_tools_agent, create_structured_chat_agent, create_json_chat_agent
from langchain_community.tools.convert_to_openai import format_tool_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain import hub
import chainlit as cl
import yfinance as yf

load_dotenv(find_dotenv())

template = """You're a financal expert. you can answer questions regarding the financial situation of companies. You also know about recent and past event, especially those related to the financial and economical world. when asked about stock price, etc. you MUST look up the current date first using the current date and time tool and use that to retrieve further temporal informations.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! 
"""
from tools import *

@cl.on_chat_start
async def on_chat_start():
    tools  = [
        Tool(
            name="stock_news_tool",
            func=lambda x: yf.Ticker(x).get_news(),
            description="Useful for when you need to find news about a company or stock. Requires a stock ticker symbol as parameter",
        ),
        Tool(
            name="stock_price_tool",
            func=lambda x: yf.Ticker(x).info,
            description="Useful for when you need to find news about a company or stock. Requires a stock ticker symbol as parameter",
        ),
        Tool(
            name="datetime",
            func=lambda x: datetime.now().strftime("%A, %B %d, %Y %H:%M:%S"),
            description="Useful for when you are need to find the current date and/or time. Call this before any other functions if you are unaware of the current date.",
        )
    ]
    tools = [search_tool, wikipedia_tool, stock_news_tool, stock_news_sentiment_tool, stock_dividend_tool, StockInfoTool(), date_tool, PythonREPLTool()]
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", template),
            ("human", "{input} \n\n {agent_scratchpad}"),
        ]
    )

    openai_functions = [format_tool_to_openai_function(t) for t in tools]
    llm_model        = ChatOpenAI(streaming=True, model="gpt-3.5-turbo", temperature=0)
    llm_with_tools   = llm_model.bind(functions=openai_functions)

    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_function_messages( x["intermediate_steps"] ),
        }
        | prompt
        | llm_with_tools
        | OpenAIFunctionsAgentOutputParser()
#        | StrOutputParser()
    )

#    prompt = hub.pull("hwchase17/openai-tools-agent")
#    agent = create_openai_tools_agent(llm_model, tools, prompt)
#    prompt = hub.pull("hwchase17/react-chat-json")
#    agent = create_json_chat_agent(llm_model, tools, prompt)
    prompt = hub.pull("hwchase17/structured-chat-agent")
    agent = create_structured_chat_agent(llm_model, tools, prompt)
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, return_intermediate_steps=False)

    cl.user_session.set("runnable", agent_executor)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")
    runnable_config = RunnableConfig(callbacks=[cl.LangchainCallbackHandler()])

    msg = cl.Message(content="")

    async for chunk in runnable.astream({ "input": message.content }, config=runnable_config):
        #await msg.stream_token(chunk)
        
        # found here: https://python.langchain.com/docs/modules/agents/how_to/streaming
        # Agent Action
        if "actions" in chunk:
            for action in chunk["actions"]:
                await msg.stream_token(
                    f"Calling Tool ```{action.tool}``` with input ```{action.tool_input}```"
                )
        # Observation
        elif "steps" in chunk:
            for step in chunk["steps"]:
                await msg.stream_token(f"Got result: ```{step.observation}```")
        # Final result
        elif "output" in chunk:
            await msg.stream_token(chunk["output"])
        else:
            raise ValueError

        #await msg.stream_token("------")

    await msg.send()