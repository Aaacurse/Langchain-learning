from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import StructuredOutputParser,ResponseSchema
from dotenv import load_dotenv
load_dotenv()


llm=HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task='text-generation'
)

model=ChatHuggingFace(llm=llm)

schema=[ResponseSchema(name="name",description="Name of the country"),
        ResponseSchema(name='capital',description="capital of the country"),
        ResponseSchema(name='currency',description="Currency of the country")
]
parser=StructuredOutputParser.from_response_schemas(schema)

template=PromptTemplate(
    template="Give the name,capital city and the currency of {country}.\n{format_instructions}",
    input_variables=['country'],
    partial_variables={'format_instructions':parser.get_format_instructions()}
)

prompt=template.invoke({'country':'India'})
chain=template|model|parser
result=chain.invoke({'country':'India'})
print(result)
print(type(result))