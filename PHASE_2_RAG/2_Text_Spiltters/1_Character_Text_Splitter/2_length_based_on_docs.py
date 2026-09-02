from langchain_classic.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


loader = PyPDFLoader(r"C:\Users\Sreenivas Bandaru\Desktop\langchain\PHASE_2_RAG\2_Text_Spiltters\1_length_based\dl-curriculum.pdf")
docs = loader.load()

spliter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=""  
)


char_list = spliter.split_documents(docs)

print("Type of char_list",type(char_list))
print("Length of the char_list = ",len(char_list))



print("Text_0 :",char_list[0])


#Text-1
print("Text_1 :",char_list[1])