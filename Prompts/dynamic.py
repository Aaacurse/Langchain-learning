from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate,load_prompt
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-2.5-flash")


st.header("Research tool")
user_paper=st.selectbox("Select research paper",["Attention is All you need","Word2Vec","BERT: Pre training of Bidirectional Transformers"])
user_style=st.selectbox("Select style",["Beginner-friendly","Mathematical","Technical","Code Oriented"])
user_length=st.selectbox("Select Length",["Short(1-2 line)","Medium(5-6 lines)","Long(8-10 lines)"])


template=load_prompt(".\\Prompts\\template.json")
prompt=template.invoke(
    {
        "paper_input":user_paper,
        "style_input":user_style,
        "length_input":user_length
    }
)

if st.button("Summarize"):
    response=model.invoke(prompt)
    st.write(response.content)
