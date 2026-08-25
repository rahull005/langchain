from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash-0731",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)


st.header("LangChain Hugging Face Chat Model")


#we are giving complete control to the user to enter any prompt and get the response from the model. The user can enter any prompt in the text input field and click on the "Send" button to get the response from the model. The response will be displayed below the input field.
user_input = st.text_input("Enter your prompt")

if st.button("Send"):
    if not user_input.strip():
        st.warning("Please enter a prompt.")
    else:
        result = model.invoke(user_input)
        st.write(result.content)

