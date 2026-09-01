from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
import os 

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_ACCESS_TOKEN")
    )


model = ChatHuggingFace(llm=llm)


loader = TextLoader(r'C:\Users\Sreenivas Bandaru\Desktop\langchain\PHASE_2_RAG\1_Document_loaders\1_text_file_loader\cricket.txt', encoding='utf-8')

docs = loader.load()

length = len(docs)

parser = StrOutputParser()

template = PromptTemplate(
    template="write a good summary on the following {topic}",
    input_variables=['topic']
)

chain = RunnableSequence(template,model,parser)

response = chain.invoke({
    'topic':docs[0].page_content
})


print(response)