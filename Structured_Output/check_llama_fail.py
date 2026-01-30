from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()

from typing import Optional,Literal
from pydantic import BaseModel,Field

llm=HuggingFaceEndpoint(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

class Review(BaseModel):
    key_themes:list[str]= Field(description="Write down all the Key themes discussed in the review")
    summary: str =Field(description="A brief summary of the review")
    sentiment:Literal["pos","neg"]= Field(default=None,description="Sentiment of the review")
    pros: Optional[list[str]] =Field(default=None,description="Write all the pros of the review")
    cons: Optional[list[str]] =Field(default=None,description="Write all the cons of the review")

structured_model=model.with_structured_output(Review)

result=structured_model.invoke(""" Pretty good quality keyboard delivered by brand. It is easy to use with complimentary battery in the packaging. The mouse is ergonomic in design which feels comfortable while using . The keyboard button is pretty durable and good function key. Also the bluetooth connectivity is very good and has dollar button in it. Overall value for money deal.""")

print(result.model_dump)