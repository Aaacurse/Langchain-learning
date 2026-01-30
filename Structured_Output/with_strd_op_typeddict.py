from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict,Annotated,Literal,Optional

model=ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class Review(TypedDict):
    key_themes: Annotated[list[str],"Write down all the Key themes discussed in the review"]
    summary: Annotated[str,"A brief summary of the review"]
    sentiment: Annotated[Literal["pos","neg"],"Sentiment of the review"]
    pros: Annotated[Optional[list[str]],"Write all the pros of the review"]
    cons: Annotated[list[str],"Write all the cons of the review"]

structured_model=model.with_structured_output(Review)

result=structured_model.invoke(""" purchased the Lapcare WL-102 2.4GHz Wireless Keyboard and Mouse for ₹650, and it is honestly the worst product I have ever bought.
After just one month of use, the keyboard and mouse stopped working suddenly. Some keys stopped responding, the mouse became unreliable, and finally the whole set became useless.
I am very unhappy and disappointed with this product. I trusted Amazon, but selling such low-quality products breaks that trust. I have lost my money and also lost my trust because of this purchase.
I do not recommend this product to anyone.
Please do not waste your money on it.
❌ Poor quality
❌ Not durable
❌ Not worth the price""")

print(result)