from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(r'C:\Users\Sreenivas Bandaru\Desktop\langchain\PHASE_2_RAG\1_Document_loaders\3_CSV_loader\Social_Network_Ads.csv')

docs = loader.load()

print(len(docs))     # returns the no.of rows
print(docs[1])       #returns each row every-time