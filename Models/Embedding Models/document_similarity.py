from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding=HuggingFaceEmbeddings(model='sentence-transformers/all-MiniLM-L6-v2')

documents = [
    "Lionel Messi is an Argentine footballer who won multiple Ballon d'Or awards and the FIFA World Cup.",
    "Python is a popular programming language used for data science, machine learning, and web development.",
    "The Taj Mahal is a white marble mausoleum located in Agra, built by Mughal emperor Shah Jahan.",
    "A neural network is a machine learning model inspired by the human brain, used in deep learning.",
    "Biryani is a spicy Indian rice dish cooked with meat or vegetables and aromatic spices."
]


query = "deep learning models inspired by the human brain"

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

similarities = cosine_similarity([query_embedding], doc_embeddings)
print(similarities)
