from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import load_prompt

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash-0731",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

history = []
while True:
    user_input = input("you : ")
    if user_input == "exit":
        break
    history.append(user_input)
    result = model.invoke(history)
    history.append(result.content)
    print("AI : ",result.content)



    #here we are storing the conversation history in a list called history. Each time the user inputs a message, it is appended to the history list. The model is then invoked with the entire conversation history, allowing it to generate a response based on the context of the previous messages. The AI's response is also appended to the history list, ensuring that the conversation context is maintained for future interactions.

    #but we have inbuilt memory in lanchain, with who are messaged, that can be managed by HumanMessage, SystemMessage, and AIMessage classes. This allows for more structured conversation management and can be useful for more complex interactions where you want to differentiate between different types of messages in the conversation.

    #next model we will see the inbuilt memory of langchain, which is more structured and can be used for more complex interactions.