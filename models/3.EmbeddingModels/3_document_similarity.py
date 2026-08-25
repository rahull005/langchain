from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

query = "What is the capital of India?"
vector = embedding.embed_query(query)    #embed_documents() method is used to convert a list

docs = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]

document_vectors = embedding.embed_documents(docs)    #embed_documents() method is used to convert a list of documents into their corresponding vector representations.

embedding_similarity = cosine_similarity([vector], document_vectors)[0]

print("Similarity Scores: ", str(embedding_similarity))


#finding index of the most similar document
index, score = sorted(list(enumerate(embedding_similarity)),key=lambda x:x[1])[-1]
print("Most Similar Document Index: ", index)
print("Similarity Score: ", score)



# printing the most similar document
result = docs[index]
print("Most Similar Document: ", result)