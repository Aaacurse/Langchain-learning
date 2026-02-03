from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser
from langchain_classic.schema.runnable import RunnableParallel

from dotenv import load_dotenv
load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task='text-generation'
)
model_gpt=ChatHuggingFace(llm=llm)
model_gemini=ChatGoogleGenerativeAI(model="gemini-2.5-flash")
str_parser=StrOutputParser()
json_parser=JsonOutputParser()

prompt1=PromptTemplate(
    template='''Generate short and simple notes from the following text:
    {text}
    ''',
    input_variables=['text']
)

prompt2=PromptTemplate(
    template='''Generate 5 short question answers from the following text:
    {text}
    ''',
    input_variables=['text'],
    # partial_variables={'format_instructions':json_parser.get_format_instructions()}
)

prompt3=PromptTemplate(
    template='Merge the provided notes and quiz into a single document\n notes:  {notes} \n quiz:{quiz}',
    input_variables=['notes','quiz']
)

parallel_chain=RunnableParallel({
    'notes': prompt1|model_gpt|str_parser,
    'quiz':prompt2|model_gemini|str_parser
})

merged_chain=prompt3|model_gemini|str_parser

chain=parallel_chain| merged_chain

text='''The **Rashtriya Swayamsevak Sangh**, commonly known as **RSS** or simply **the Sangh**, is one of the most influential socio-cultural organizations in India. Founded in **1925** in Nagpur by **Dr. Keshav Baliram Hedgewar**, the RSS emerged during a period of intense political, social, and cultural churn under British colonial rule. Its primary stated objective was to organize Hindu society, build character, and foster a sense of national unity rooted in Indian cultural values.

At its ideological core, the RSS believes that India’s national identity is deeply intertwined with its ancient civilization, traditions, and cultural continuity. The organization promotes the idea of *cultural nationalism*, often described as **Hindutva**, which views India as a civilizational nation rather than merely a political construct. According to RSS thought, strengthening society culturally and morally is a prerequisite for a strong, unified nation.

One of the most distinctive aspects of the RSS is its **organizational structure and discipline**. The basic unit of the Sangh is the **shakha**, a daily or weekly gathering where members (called *swayamsevaks*) meet, usually in open grounds. These shakhas involve physical exercises, drills, games, patriotic songs, discussions, and ideological training. The emphasis is not only on physical fitness but also on discipline, teamwork, leadership, and a sense of service. Participation is voluntary, and the organization traditionally does not maintain formal membership registers.

The RSS describes itself as a **non-political organization**, focusing on social and cultural work rather than electoral politics. However, over time, it has built a vast ecosystem of affiliated organizations, collectively referred to as the **Sangh Parivar**. These include groups working in education, labor unions, tribal welfare, women’s empowerment, religious outreach, student movements, and politics. The most prominent political organization associated with the Sangh Parivar is the Bharatiya Janata Party (BJP), though the RSS maintains that it does not directly control political decision-making.

Social service has been a major pillar of RSS activity. The organization and its affiliates are active in **disaster relief**, **rural development**, **healthcare**, **education**, and **tribal welfare**, especially in remote and underserved regions. During natural calamities such as floods, earthquakes, and pandemics, RSS volunteers are often visible on the ground, providing food, medical assistance, and logistical support. Supporters cite this grassroots service work as evidence of the Sangh’s nation-building role beyond ideology.

At the same time, the RSS has been a subject of **significant debate and controversy**. Critics argue that its interpretation of nationalism and cultural identity can marginalize religious minorities, particularly Muslims and Christians. The organization has also been accused by opponents of promoting majoritarian views and influencing state institutions indirectly through its wide network. The RSS, however, consistently rejects allegations of intolerance, stating that it respects all religions while emphasizing India’s indigenous cultural ethos.

Historically, the RSS has had a complex relationship with the Indian state. It was banned multiple times—most notably after the assassination of Mahatma Gandhi in 1948 (a ban later lifted when no organizational culpability was proven) and during the Emergency period (1975–77). These episodes played a crucial role in shaping the Sangh’s self-image as a disciplined organization resilient under pressure and committed to long-term societal work rather than short-term political gains.

In contemporary India, the RSS continues to exert substantial **ideological and organizational influence**. Its leadership, headed by the *Sarsanghchalak*, emphasizes patience, long-term vision, and societal transformation through incremental change. The Sangh’s supporters see it as a stabilizing force preserving cultural heritage and national unity, while critics view it as a powerful ideological movement whose influence must be critically examined in a pluralistic democracy.

Overall, the Rashtriya Swayamsevak Sangh occupies a **central and complex place in modern Indian society**. Whether seen as a cultural revivalist organization, a disciplined volunteer movement, or an ideological force shaping national discourse, its impact on India’s social and political landscape over the past century is undeniable—and continues to be a subject of active discussion, support, and criticism.
'''
result=chain.invoke({'text':text})
print(result)
chain.get_graph().print_ascii()