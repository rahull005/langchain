from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

loader = PyPDFLoader(r'C:\Users\Sreenivas Bandaru\Desktop\langchain\PHASE_2_RAG\1_Document_loaders\2_Pdf_loader\dl-curriculum.pdf')

docs = loader.load()

print(type(docs))
print(len(docs))             #len(pdf) = return no.of pages


print(docs[0].page_content)
print(docs[1].metadata)