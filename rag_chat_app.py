import streamlit as st

from rag import setup_rag, get_model, setup_retriever

st.set_page_config(page_title="RAG Chat")
st.title("RAG Chat")


MODEL_DEFAULTS = {'temperature': 0.3, 'top_k': 25, 'top_p': 0.7, 'num_predict': 1024 }
MODEL_PARAMS = MODEL_DEFAULTS.keys()

for mp in MODEL_PARAMS:
    if mp not in st.session_state:
        st.session_state[mp] = MODEL_DEFAULTS[mp]


st.sidebar.header("Model Parameters")
with st.sidebar:
    st.slider("Temperature", min_value=0.0, max_value=1.0, step=0.01, key="temperature")
    st.number_input("Top-K", min_value=1, max_value=50, key="top_k")
    st.slider("Top-P", min_value=0.0, max_value=1.0, key="top_p")
    st.number_input("Max New Tokens", min_value=25, max_value=4096, key="num_predict")   
    

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


def ask_question(chain, query):
    response = chain.invoke(
        {"question": query},
        # this requires a session ID on input.
        # this is for tracking per-session things which we don't have.
        # so it can just be a placeholder.
        config={"configurable": {"session_id": "foo"}}
    )
    return response


@st.cache_resource
def create_model(model_params):
    return get_model(**model_params)


@st.cache_resource
def create_retriever():
    retriever, vs = setup_retriever()
    return retriever


@st.cache_resource
def setup():
    st.write("Loading...")
    rag_chain = setup_rag()
    return rag_chain


def run():
    model_params = {}
    for mp in MODEL_PARAMS:
        if mp in st.session_state:
            model_params[mp] = st.session_state[mp]

    st.write("Model params are:")
    st.write(model_params)
    
    model = create_model(model_params)
    retriever = create_retriever()
    rag_chain = setup_rag(model, retriever)
    st.write("Ready.")

    st.subheader("Ask me a question. I will keep track of our discussion:")
    show_ui(rag_chain, "How can I help you?")


run()
