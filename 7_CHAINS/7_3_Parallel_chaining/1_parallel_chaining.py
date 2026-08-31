from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
import os

load_dotenv()

#first-model
llm1 = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_ACCESS_TOKEN")
    )

model = ChatHuggingFace(llm=llm1)


# #second-model
# llm2 = HuggingFaceEndpoint(
#     repo_id="deepseek-ai/DeepSeek-V4-Flash-0731",
#     task="text-generation",
# )

# model2 = ChatHuggingFace(llm=llm2)

#now we have to create 2 chains and finally create a merger chain and combine them
#first we have to create 2 templates, one is to give the summary on the topic.
#the other template is to make a quiz on the topic
#Other template is to combine/merge


template1 = PromptTemplate(
    template="explain in detail about the following {topic}",
    input_variables=['topic']
)

template2 = PromptTemplate(
    template='Generate 5 short question answers from the following text \n {topic}',
    input_variables=['topic']
)



template3 = PromptTemplate(
    template="combine the {notes} and the {quiz} into one single doc",
    input_variables=['notes','quiz']
)


#parser
parser = StrOutputParser()

parallel_chains = RunnableParallel({
    'notes': template1 | model | parser,
    'quiz' : template2 | model | parser
})


merge_chain = template3 | model | parser

final_chain = parallel_chains | merge_chain


#invoking the model
response = final_chain.invoke({'topic':'Chat-GPT'})


print(response)
print(final_chain.get_graph().print_ascii())