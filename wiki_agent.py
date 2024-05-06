from langchain import hub
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActJsonSingleInputOutputParser
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.llms import Ollama
from langchain.agents import create_json_chat_agent, AgentExecutor


# we also want Beautiful soup installed.
# pip install beautifulsoup4
from langchain_core.tools import render_text_description

MODEL_ID="mistral"

# Remember that Ollama must be running.
model = Ollama(
   model=MODEL_ID, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)

wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=3000))

# try out our tool
wiki_tool.run("endoscopy")

tools = [wiki_tool]


# setup ReAct style prompt
prompt = hub.pull("hwchase17/react-json")
prompt = prompt.partial(
    tools=render_text_description(tools),
    tool_names=", ".join([t.name for t in tools]),
)

# Tells it to stop when it sees "Observation"
chat_model_with_stop = model.bind(stop=["\nObservation"])
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
    handle_parsing_errors=True, # make it keep trying if it has an error.
    verbose=True)

agent_executor.invoke({"input": "When did Washington become a state?"})
