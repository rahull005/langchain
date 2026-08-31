from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel,RunnableSequence,RunnablePassthrough,RunnableBranch
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_ACCESS_TOKEN")
    )


model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Summarize the following text \n {text}',
    input_variables=['text']
)

 
parser = StrOutputParser()

#chaining
# chain = prompt1 | model | parser
chain = RunnableSequence(prompt1,model,parser)


#condtional-chaining
conditional_chain = RunnableBranch(
    (lambda x: len(x.split()) > 300, RunnableSequence(prompt2,model,parser)), #prompt2 | model | parser
    RunnablePassthrough()
)

#final-chain
final_chain = RunnableSequence(chain,conditional_chain)


#invoke
response = final_chain.invoke({'topic':'USA tarrifs'})

print(response)