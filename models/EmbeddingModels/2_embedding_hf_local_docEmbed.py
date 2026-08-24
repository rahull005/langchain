from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]

vector = embedding.embed_documents(documents)    #embed_documents() method is used to convert a list of documents into their corresponding vector representations.
                                                 #embed_query() method is used to convert a single query into its corresponding vector representation.
print(str(vector))