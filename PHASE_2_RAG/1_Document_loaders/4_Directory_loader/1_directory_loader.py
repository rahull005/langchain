from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

#combines all the pdf pages in the folder
#assume we have (3-pdf's) of pages(20,30,40), then after loading it gives us of length = 90
loader = DirectoryLoader(
    path=r'C:\Users\Sreenivas Bandaru\Desktop\langchain\PHASE_2_RAG\1_Document_loaders\4_Directory_loader\books',
    glob='*.pdf',              # tells to load all the files with extension (.pdf)
    loader_cls=PyPDFLoader     # this is the loader_class
)

docs = loader.load()

print("length of the docs = ",len(docs))

print("first_doc_page-content",docs[325].page_content)

print(type(docs))
# for document in docs:
#     print(document.metadata)