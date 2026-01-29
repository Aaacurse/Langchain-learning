from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage,HumanMessage,SystemMessage
from dotenv import load_dotenv
load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-2.5-flash")

chat_history=[
    SystemMessage(content="You are a helpful assistant")
]
while True:
    user_input=input("You: ")
    if(user_input.lower()=="exit"):
        break
    
    chat_history.append(HumanMessage(content=user_input))


    result=model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print(f"AI: {result.content}")

print(chat_history)