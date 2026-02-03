from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.schema.runnable import RunnableParallel,RunnableBranch,RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv
from pydantic import BaseModel,Field
from typing import Literal
load_dotenv()

class Feedback(BaseModel):
    sentiment: Literal["positive","negative"]=Field(description="Give the sentiment of the feedback")

pyparser=PydanticOutputParser(pydantic_object=Feedback)

llm=HuggingFaceEndpoint(
    repo_id='openai/gpt-oss-20b',
    task='text-generation'
)
model=ChatHuggingFace(llm=llm)
parser=StrOutputParser()
prompt1=PromptTemplate(
    template="Classify the sentiment of the following text into Positive or Negative:\n  {feedback}\n {format_instruction}",
    input_variables=['feedback'],
    partial_variables={'format_instruction':pyparser.get_format_instructions()}
)

prompt2=PromptTemplate(
    template="Write an appropriate response to this positive feedback: \n {feedback}",
    input_variables=['feedback']
)

prompt3=PromptTemplate(
    template="Write an appropriate response to this positive feedback: \n {feedback}",
    input_variables=['feedback']
)

classifier_chain=prompt1|model|pyparser

branch_chain=RunnableBranch(
    (lambda x: x.sentiment=='positive',prompt2|model|parser),
    (lambda x: x.sentiment=='negative',prompt2|model|parser),
    RunnableLambda(lambda x: "could not find sentiment")
)

chain=classifier_chain|branch_chain

print(chain.invoke({'feedback':'This is a wonderful product'}))