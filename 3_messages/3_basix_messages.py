from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash-0731",
    task="text-generation",
)   

model = ChatHuggingFace(llm=llm)

messages =[
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Hello, how are you?")
]


result = model.invoke(messages)
messages.append(AIMessage(content=result.content))
print("AI : ",result.content)