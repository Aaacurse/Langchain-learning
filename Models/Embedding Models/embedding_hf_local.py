from langchain_huggingface import HuggingFaceEmbeddings

embeddding=HuggingFaceEmbeddings(model='sentence-transformers/all-MiniLM-L6-v2')

text="Delhi is the capital of India"

result=embeddding.embed_query(text)

print(result)