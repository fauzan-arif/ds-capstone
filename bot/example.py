from dotenv import find_dotenv, load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig

import chainlit as cl

load_dotenv(find_dotenv())


@cl.on_chat_start
async def on_chat_start():
    model  = ChatOpenAI(streaming=True)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You're a financal expert. you can answer questions regarding the financial situation of companies. when asked about stock price, etc. you MUST look up the current date first using the current date and time tool and use that to retrieve further temporal informations. You have access to a python REPL, which you can use to execute python code. you have access to yahoo finance informations using the stock price and the stock information tool. you also have access to web search and wikipedia.",
            ),
            ("human", "{question}"),
        ]
    )
    runnable = prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")
    runnable_config = RunnableConfig(callbacks=[cl.LangchainCallbackHandler()])

    msg = cl.Message(content="")

    async for chunk in runnable.astream( {"question": message.content}, config=runnable_config):
        await msg.stream_token(chunk)

    await msg.send()