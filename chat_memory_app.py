import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.llms import Ollama
from langchain_core.runnables.history import RunnableWithMessageHistory

st.set_page_config(page_title="LLM Chat with Memory")
st.title("LLM Chat with Memory")

# Feel free to try other models.
MODEL_ID = "mistral"

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


def create_memory_chain(llm, base_chain, chat_memory):
    # This example uses the model itself to compress the chat history so it can be
    # included in the context for the next question without exceeding the context window.
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    runnable = contextualize_q_prompt | llm | base_chain

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        # again, we are not really using the session ID here.
        return chat_memory

    with_message_history = RunnableWithMessageHistory(
        runnable,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    return with_message_history


def get_chain():
    model = get_model()
    base = basic_chain(model)
    # Now we are adding memory of the conversation.
    chain = create_memory_chain(model, base, chat_memory=StreamlitChatMessageHistory(key="langchain_messages"))

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



def run():
    ready = True

    if ready:
        chain = get_chain()
        st.subheader("Ask me a question. I will keep track of our discussion:")
        show_ui(chain, "How can I help you?")
    else:
        st.stop()


run()
