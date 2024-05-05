import streamlit as st
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama

st.set_page_config(page_title="LLM Chat")
st.title("LLM Chat")

# Feel free to try other models.
MODEL_ID = "gemma:2b"

# Try experimenting with this to see how you can change the app's behavior.
PROMPT = "You are a helpful AI assistant. Answer the user's request in a factual but friendly manner."


# This uses Ollama so remember to make sure it is running.
def get_model():
    llm = Ollama(
        model=MODEL_ID,
        # You can take out the callback manager to stop printing things to the terminal stdout.
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )
    return llm

def basic_chain(model):
    prompt = ChatPromptTemplate.from_messages([
        ("system", PROMPT),
        ("human", "My research request is {input}"),
    ])
    chain = prompt | model
    return chain


def ask_question(chain, query):
    response = chain.invoke(
        {"input": query},
        # this requires a session ID on input.
        # this is for tracking per-session things which we don't have.
        # so it can just be a placeholder.
        config={"configurable": {"session_id": "foo"}}
    )
    return response


def show_ui(qa, prompt_to_user="How may I help you?"):
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": prompt_to_user}]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User-provided prompt
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = ask_question(qa, prompt)
                st.markdown(response)
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)


def get_chain():
    model = get_model()
    chain = basic_chain(model)
    return chain


def run():
    ready = True

    if ready:
        chain = get_chain()
        st.subheader("Ask me a question:")
        show_ui(chain, "How can I help you?")
    else:
        st.stop()


run()
