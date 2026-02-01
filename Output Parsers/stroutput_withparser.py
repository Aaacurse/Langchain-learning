from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
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
parser=StrOutputParser()

chain=template1 | model | parser |template2 | model |parser

result=chain.invoke({'topic':'Black Hole'})

print(result)