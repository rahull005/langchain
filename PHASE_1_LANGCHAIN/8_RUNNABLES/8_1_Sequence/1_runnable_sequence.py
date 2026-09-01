from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_ACCESS_TOKEN")
    )

# 1 - model
model = ChatHuggingFace(llm=llm)


# 2 - Prompt-template 
template1 = PromptTemplate(
    template="write a joke on the following {topic}",
    input_variables=['topic']
)

template2 = PromptTemplate(
    template='Explain the following joke - {text}',
    input_variables=['text']
)



# 3 - Parser
parser = StrOutputParser()

# 4 - chain
chain = RunnableSequence(template1,model,parser,template2,model,parser)

# 5 - reponse
response = chain.invoke({'topic':'cricket'})

#print
print(response)
