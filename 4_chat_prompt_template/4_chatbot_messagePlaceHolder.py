from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash-0731",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful {domain} expert."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

domain = input("Domain: ")

chat_history = []


while True:
    user_input = input("Question: ")
    if user_input == "exit":
        break

    prompt = template.invoke({
        "domain": domain,
        "question": user_input,
        "chat_history": chat_history,
    })

    result = model.invoke(prompt)
    chat_history.append(
        HumanMessage(content=user_input)
    )
    chat_history.append(
        AIMessage(content=result.content)
    )
    print("AI : ", result.content)

