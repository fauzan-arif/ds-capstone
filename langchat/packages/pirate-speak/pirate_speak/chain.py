from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Translate user input into pirate speak",
        ),
        ("human", "{text}"),
    ]
)
_model = ChatOpenAI(openai_api_key="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")

# if you update this, you MUST also update ../pyproject.toml
# with the new `tool.langserve.export_attr`
chain = _prompt | _model
