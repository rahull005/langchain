from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel,RunnableSequence,RunnablePassthrough,RunnableBranch,RunnableLambda
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

#we we use this later to count the letters in the output

template = PromptTemplate(
    template = "Write a joke on the following {topic}",
    input_variables=['topic']
)

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(template,model,parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'length':RunnableLambda(lambda x : len(x.split()))
})



final_chain = RunnableSequence(joke_gen_chain,parallel_chain)

response = final_chain.invoke({'topic':'ai models'})

print(response)
print("=====================================")
print("Joke = ",response['joke'])
print("=====================================")
print("Lenght = ",response['length'])

