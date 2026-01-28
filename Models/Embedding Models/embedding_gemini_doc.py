from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding=GoogleGenerativeAIEmbeddings(model="gemini-embedding-001",output_dimensionality=32)

documents=["Delhi is capital of India","Kolkata is capital of Bengal","Paris is capital of France"]

result=embedding.embed_documents(documents)

print(len(result))