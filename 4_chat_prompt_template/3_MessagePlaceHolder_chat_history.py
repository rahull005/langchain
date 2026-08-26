from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash-0731",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}")
])

chat_history = []

with open("/Users/rahulrap/Documents/langchain/4_chat_prompt_template/4_chat_history.py") as f:
    chat_history.append(f.readline())

print("Chat History:", chat_history)

prompt = template.invoke({
    "chat_history": chat_history,
    "input": "What is the capital of France?"
})

print("Prompt:", prompt)