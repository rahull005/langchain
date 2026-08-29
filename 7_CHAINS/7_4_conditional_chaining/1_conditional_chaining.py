from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableLambda,RunnableBranch
from pydantic import BaseModel,Field
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from typing import Literal

load_dotenv()

#first-model
llm1 = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task="text-generation"
    )

model = ChatHuggingFace(llm=llm1)

#here we are using pydantic Literal because,
#the llm wont give the exact answer we want, but if we use the lliteral
# it 100% gives the solution that we are excepecting

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')

pydantic_parser = PydanticOutputParser(pydantic_object=Feedback)

template1 = PromptTemplate(
    template="Classify the sentiment of the following feedback text into postive or negative \n {feedback} \n {format_instruction}",
    input_variables=['feedback'],
    partial_variables={'format_instruction':pydantic_parser.get_format_instructions()}
)

#from the above template we get the output as postive/negative
#based on that we will use the condition, if positive we go for one flow, else, other flow

pos_neg_chain = template1 | model | pydantic_parser

# pos_neg_response = pos_neg_chain.invoke({'feedback':'its a great phone'})

# print(pos_neg_response.sentiment)



#now based on the feedback we will decide the chain
template2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

template3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)


str_parser = StrOutputParser()


branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', template2 | model | str_parser),
    (lambda x:x.sentiment == 'negative', template3 | model | str_parser),
    RunnableLambda(lambda x: "could not find sentiment")
)

chain = pos_neg_chain | branch_chain

print(chain.invoke({'feedback': 'This is a worst phone'}))

chain.get_graph().print_ascii()