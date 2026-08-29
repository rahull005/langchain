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

template = PromptTemplate(
    template="give me 5 lines overview on the following {topic}",
    input_variables=['topic']
)

parser = StrOutputParser()

chain = template | model | parser

response = chain.invoke({'topic':'neural networks'})

print(response)

# see the chain
print(chain.get_graph().print_ascii())