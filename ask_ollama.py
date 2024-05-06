from langchain_community.llms.ollama import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

MODEL_ID = "gemma:2b"
# MODEL_ID="mistral"

BASIC_PROMPT = """You are a helpful AI assistant. Answer the user's request accurately."""


# This uses Ollama so remember to make sure it is running.
def get_model(temperature=0.1, top_k=50, top_p=0.7, num_predict=1024):
    llm = Ollama(
        model=MODEL_ID,
        # You can take out the callback manager to stop printing things to the terminal stdout.
        # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])

    )

    # These parameters are documented for Ollama at
    # https://github.com/ollama/ollama/blob/main/docs/modelfile.md
    llm = llm.bind(temperature=temperature, top_k=top_k, top_p=top_p, num_predict=num_predict)

    return llm


def basic_chain(model):
    prompt = ChatPromptTemplate.from_messages([
        ("system", BASIC_PROMPT),
        ("human", "{input}"),
    ])
    chain = prompt | model
    return chain


def ask_chain(chain, query):
    print(f"Query: {query}\nThinking...")
    response = chain.invoke(query)

    print(f"Response:\n{response}")


def setup_chain():
    model = get_model()
    output_parser = StrOutputParser()
    chain = basic_chain(model) | output_parser
    return chain


def main():
    chain = setup_chain()

    while True:
        user_input = input("Ask me a question or 'quit' to exit: ")
        if user_input == "quit":
            break
        if user_input:
            ask_chain(chain, user_input)


if __name__ == '__main__':
    main()
