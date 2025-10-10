from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from langchain.schema import SystemMessage, HumanMessage

load_dotenv()

llm = AzureChatOpenAI(
    azure_deployment="gpt-35-turbo",  # or your deployment
    api_version="2025-01-01-preview",  # or your api version
)

user_input = input("you: ")

messages = [
    SystemMessage(content="you are a personal assistant you remember everything"),
    HumanMessage(content=user_input)
]

response = llm(messages)
print(response.content)
