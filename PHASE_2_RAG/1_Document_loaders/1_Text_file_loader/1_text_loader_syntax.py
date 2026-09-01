from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()


loader = TextLoader(r'C:\Users\Sreenivas Bandaru\Desktop\langchain\PHASE_2_RAG\1_Document_loaders\1_text_file_loader\cricket.txt', encoding='utf-8')

docs = loader.load()

print("Type of docs is = ",type(docs))       #returns the list of Documents


#print 1st doc and doc type
print("type of first file in doc_list = ",type(docs[0]))


#print the first doc content and metadata
print("Page Content = ",docs[0].page_content)

print("metadata = ",docs[0].metadata)


#length
print("length of docs = ",len(docs))