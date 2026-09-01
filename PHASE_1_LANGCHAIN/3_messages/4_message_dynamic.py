from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage,AIMessage


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash-0731",
    task="text-generation",
)  

model = ChatHuggingFace(llm=llm)

user_input = input("You: ")

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=user_input),
]

result = model.invoke(messages)
messages.append(AIMessage(content=result.content))
print("AI:", result.content)