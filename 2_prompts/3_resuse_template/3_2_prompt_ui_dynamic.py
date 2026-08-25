from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import load_prompt
from pathlib import Path

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash-0731",
    task="text-generation",
)   


model = ChatHuggingFace(llm=llm)

st.header("LangChain Hugging Face Chat Model")

#single-select front-end with dynamic prompt template
paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )


# Step 1: The template was created in 3_1_prompt_template.py.
# Step 2: 3_1_prompt_template.py saved that template as prompt_template.json.
# Step 3: Load the saved JSON file from this same folder.
template_path = Path(__file__).resolve().parent / "prompt_template.json"
template = load_prompt(str(template_path))

# Step 4: Fill the template with the values selected in the Streamlit interface.
prompt = template.invoke({
    "paper_input": paper_input,
    "style_input": style_input,
    "length_input": length_input
})

# Step 5: Send the completed prompt to the chat model and display its response.
if st.button("Generate Explanation"):
    result = model.invoke(prompt)
    st.write(result.content)