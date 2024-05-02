
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.llms import Ollama

PROMPT = """You are a poor street urchin from a Charles Dickens novel.
Respond to the user's query in Cockney rhyming slang, and always conclude with an offer to help for a fee.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", PROMPT),
    ("human", "{input}"),
])
model = Ollama(model="gemma")

output_parser = StrOutputParser()
chain = prompt | model | output_parser
response = chain.invoke("Can you tell me how to get to Big Ben from Victoria Station?")
#response = chain.invoke("Where can I get a proper English breakfast?")
print(response)


