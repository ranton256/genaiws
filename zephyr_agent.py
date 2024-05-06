import os

import toml
from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import (
    ReActJsonSingleInputOutputParser,
)
from langchain.chains.llm_math.base import LLMMathChain

from langchain.memory import ChatMessageHistory
from langchain.tools.render import render_text_description
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.tools import WikipediaQueryRun, PubmedQueryRun
from langchain_community.tools.semanticscholar.tool import SemanticScholarQueryRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper, ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import Tool
from langchain_community.tools.wikidata.tool import WikidataAPIWrapper, WikidataQueryRun


with open('secrets.toml', 'r') as f:
    config = toml.load(f)
    huggingfacehub_api_token = config['HUGGINGFACEHUB_API_TOKEN']
    os.environ['HUGGINGFACEHUB_API_TOKEN'] = huggingfacehub_api_token

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    max_new_tokens=1024,
    temperature=0.1,
    top_k=5,
)

chat_model = ChatHuggingFace(llm=llm)

search_wrapper = DuckDuckGoSearchAPIWrapper(region="en-us", time="d", max_results=3)
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=False)
arxiv = ArxivAPIWrapper()

duckduckgo_tool = Tool(
    name="Search",
    func=search_wrapper.run,
    description="DuckDuckGo web search. This is useful for recent information or events. You should ask targeted "
                "questions "
)

calc_tool = Tool(
    name="Calculator",
    func=llm_math_chain.run,
    description="useful for when you need to answer questions about math"
)

# pip install arxiv
arxiv_tool = Tool(
    name="Arxiv",
    func=arxiv.run,
    description="useful when you need an answer about encyclopedic general knowledge"
)

# pubmed tool requires xmltodict package
# pip install xmltodict
pubmed_tool = PubmedQueryRun()

# pip install --upgrade --quiet  wikibase-rest-api-client mediawikiapi
wikidata_tool = WikidataQueryRun(api_wrapper=WikidataAPIWrapper())

# pip install wikipedia
wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=3000))

# pip install semanticscholar
semantic_scholar_tool = SemanticScholarQueryRun()

tools = [duckduckgo_tool, calc_tool, arxiv_tool, pubmed_tool, wikidata_tool, wikipedia_tool, semantic_scholar_tool]

# setup ReAct style prompt
prompt = hub.pull("hwchase17/react-json")
prompt = prompt.partial(
    tools=render_text_description(tools),
    tool_names=", ".join([t.name for t in tools]),
)

chat_model_with_stop = chat_model.bind(stop=["\nObservation"])
agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
        }
        | prompt
        | chat_model_with_stop
        | ReActJsonSingleInputOutputParser()
)

memory = ChatMessageHistory(session_id="test-session")

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    return_intermediate_steps=True,
    verbose=True)

agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    # In a real production case we would need real session ID's but ignoring it here.
    lambda session_id: memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)


def ask_agent(query):
    print(f"QUERY: {query}")
    response = agent_with_chat_history.invoke(
        {"input": query},
        config={"configurable": {"session_id": "<foo>"}},
    )
    print("\nRESPONSE:\n", response['output'])


def main():
    PROMPT = """I want you to act as a research assistant. Research the 
        user's request using reliable sources, organize the material in a well-structured way.
        """

    # examples
    # ask_agent("What is the cube root of 67?")
    # ask_agent("What are the five most cited papers related to natural language processing since 2016?")

    while True:
        user_input = input("Ask me a question or 'quit' to exit: ")
        if user_input == "quit":
            break
        if user_input:
            ask_agent(f"{PROMPT} {user_input}")


if __name__ == "__main__":
    main()
