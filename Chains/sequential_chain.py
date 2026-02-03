from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

template1=PromptTemplate(
    template='''Generate a detailed report on {topic}''',
    input_variables=['topic']
)

template2=PromptTemplate(
    template='''Generate a 5 pointer summary on the report given below: 
    {response}
    ''',
    input_variables=['response']
)
llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task='text-generation'
)

model=ChatHuggingFace(llm=llm)
parser=StrOutputParser()

chain=template1|model|parser|template2|model|parser

result=chain.invoke({'topic':'Black hole'}) 

print(result)