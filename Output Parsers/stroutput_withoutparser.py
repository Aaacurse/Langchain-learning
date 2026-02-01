from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task='text-generation'
)

model=ChatHuggingFace(llm=llm)

#1st Prompt-> Detailed report 
template1=PromptTemplate(
    template="""Write a detailed report on {topic}""",
    input_variables=['topic'],
    validate_template=True
)

#2nd Prompt -> 5 lines
template2=PromptTemplate(
    template="Write a 5 line summary on the following text. \n {text}",
    input_variables=['text'],
    validate_template=True
)

prompt1=template1.invoke({"topic":"Machine Learning"})
response1=model.invoke(prompt1)
prompt2=template2.invoke(response1.content)

result=model.invoke(prompt2)
print(result.content)