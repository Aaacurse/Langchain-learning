from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
load_dotenv()


llm=HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task='text-generation'
)

model=ChatHuggingFace(llm=llm)
parser=JsonOutputParser()

template=PromptTemplate(
    template="Give the name,capital city and the currency of India,USA,France and China.\n{format_instructions}",
    input_variables=[],
    partial_variables={'format_instructions':parser.get_format_instructions()}
)

prompt=template.format()
chain=template|model|parser
result=chain.invoke({})
print(result)
print(type(result))