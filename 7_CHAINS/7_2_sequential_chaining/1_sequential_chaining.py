from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task="text-generation"
    )

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template="Geneate the summary on the follwing {topic}",
    input_variables=['topic']
)


template2 = PromptTemplate(
    template="give five points on the {text}",
    input_variables=['text']
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

response = chain.invoke({'topic':'Convolutional Neural Networks'})

print(response)



