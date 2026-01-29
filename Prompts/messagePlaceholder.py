from langchain_core.prompts import MessagesPlaceholder,ChatPromptTemplate
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage

chat_message=ChatPromptTemplate([
    ("system","You are a helful cutomer support agent"),
    MessagesPlaceholder(variable_name='chat_history'),
    ("human",'{query}')
])
chat_history=[]

with open(".\\Prompts\\chat_history.txt","r") as f:
    chat_history.extend(f.readlines())


prompt=chat_message.invoke({
    "chat_history":chat_history,
    "query":"offside"
})

print(prompt)