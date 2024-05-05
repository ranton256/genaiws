import os
from pathlib import Path

from PyPDF2 import PdfReader
from langchain import hub
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_community.vectorstores import Chroma
from langchain_core.messages.base import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda


# setup constants
USE_CUSTOM_PROMPT = False

# MODEL_ID="gemma:2b"
# MODEL_ID="mistral"
MODEL_ID="gemma"
# MODEL_ID = "llama3"

# EMBEDDING_MODEL_ID = 'nomic-embed-text'
EMBEDDING_MODEL_ID = 'snowflake-arctic-embed'


# This uses Ollama so remember to make sure it is running.
def get_model():
    llm = Ollama(
        model=MODEL_ID,
        # You can take out the callback manager to stop printing things to the terminal stdout.
        # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )
    return llm


BASIC_PROMPT = """You are a helpful AI assistant. Answer the user's request accurately."""

RESEARCHER_PROMPT = """I want you to act as an academic research assistant. You are responsible for researching the 
user's research request and presenting the findings in a paper or article form. Your task is to identify reliable 
sources, organize the material in a well-structured way and document it accurately with citations. 
If you don't know the answer, just say that you don't know.
"""

RESEARCHER_RAG_PROMPT = """I want you to act as an academic research assistant. You are responsible for researching the 
user's research request and presenting the findings in a paper or article form. Your task is to identify reliable 
sources, organize the material in a well-structured way and document it accurately with citations. 
If you don't know the answer, just say that you don't know.

Use the following pieces of retrieved context as primary sources for your research.

"""


def basic_chain(model, prompt=None):
    if not prompt:
        prompt = ChatPromptTemplate.from_messages([
            ("system", BASIC_PROMPT),
            ("human", "My research request is {input}"),
        ])
    chain = prompt | model
    return chain


# If you want to use the current directory instead of the directory
# of this source file, you can use "./" instead of __file__
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
CONTENT_DIR = os.path.join(ROOT_PATH, "gi_guideline_docs")


def list_data_files(data_dir=CONTENT_DIR):
    pdf_paths = Path(data_dir).glob('**/*.pdf')
    txt_paths = Path(data_dir).glob('**/*.txt')
    paths = list(pdf_paths)
    paths.extend(txt_paths)
    print(f"Found {len(paths)} documents in {data_dir}")
    return paths


def load_data_files(paths):
    all_docs = []
    for path in paths:
        print(f"Loading {path}")
        filename = os.path.basename(path)
        if filename.lower().endswith('pdf'):
            docs = load_from_pdf(path)
        elif filename.lower().endswith('txt'):
            loader = TextLoader(path)
            docs = loader.load()
        else:
            print(f"WARNING! Do not know how to load file {path}")
            docs = []
        all_docs.extend(docs)
    return all_docs


def load_from_pdf(path):
    title = os.path.basename(path)
    docs = []
    with open(path, "rb") as f:
        pdf_reader = PdfReader(f)
        for num, page in enumerate(pdf_reader.pages):
            page = page.extract_text()
            doc = Document(page_content=page, metadata={'title': title, 'page': (num + 1)})
            docs.append(doc)

    return docs


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# TODO: we should not need this anymore (probably)
def get_question(input_obj):
    if not input_obj:
        return None
    elif isinstance(input_obj, str):
        return input_obj
    elif isinstance(input_obj, dict) and 'question' in input_obj:
        return input_obj['question']
    elif isinstance(input_obj, BaseMessage):
        return input_obj.content
    else:
        raise Exception("string or dict with 'question' key expected as RAG chain input.")


def make_rag_chain(model, retriever, rag_prompt=None):
    # We will use a prompt template from langchain hub.
    if not rag_prompt:
        rag_prompt = hub.pull("rlm/rag-prompt")

    # Use RunnablePassthrough to add some custom processing into our chain.
    rag_chain = (
            {
                "context": RunnableLambda(get_question) | retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | rag_prompt
            | model
    )

    return rag_chain


def filename_from_url(url):
    filename = url.split("/")[-1]
    return filename


def split_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False)

    contents = docs
    if docs and isinstance(docs[0], Document):
        contents = [doc.page_content for doc in docs]

    texts = text_splitter.create_documents(contents)
    n_chunks = len(texts)
    print(f"Split into {n_chunks} chunks")
    return texts


def create_vector_db(chunks):
    print(f"Embedding {len(chunks)} chunks into vector DB")
    embedding_function = OllamaEmbeddings(model=EMBEDDING_MODEL_ID)
    vectorstore = Chroma.from_documents(chunks, embedding=embedding_function)
    return vectorstore


def sim_search(vs, query):
    print(f"Query: {query}\n")
    results = vs.similarity_search(query)
    for i, text in enumerate(results):
        content = text.page_content
        print(f"Result {i + 1}: {content}\n")


def ask_chain(chain, query):
    print(f"Query: {query}\nThinking...")
    response = chain.invoke(query)
    print(f"Response:\n{response}")


def setup_retriever():
    doc_paths = list_data_files()
    if not doc_paths:
        print(f"No documents found!")
        return None
    docs = load_data_files(doc_paths)
    texts = split_documents(docs)
    vs = create_vector_db(texts)
    retriever = vs.as_retriever()
    return retriever



def setup_rag():
    model = get_model()
    output_parser = StrOutputParser()

    retriever = setup_retriever()
    if retriever:
        if USE_CUSTOM_PROMPT:
            rag_prompt = ChatPromptTemplate.from_messages([
                ("system", RESEARCHER_RAG_PROMPT),
                ("human", "My research request is {question}"),
            ])
            chain = make_rag_chain(model, retriever, rag_prompt) | output_parser
        else:
            chain = make_rag_chain(model, retriever) | output_parser
    else:
        print(f"Could not create retriever. Falling back to base model.")
        if USE_CUSTOM_PROMPT:
            chain = basic_chain(model, RESEARCHER_PROMPT) | output_parser
        else:
            chain = basic_chain(model) | output_parser

    return chain


def main():
    rag_chain = setup_rag()

    # example_q = "What criteria are used to determine which patients to screen for esophageal adenocarcinoma?"
    # print("I will use the provided documents to answer your questions.")
    # print("For example: ", example_q)
    # ask_chain(rag_chain, example_q)

    while True:
        user_input = input("Ask me a question about the documents or 'quit' to exit: ")
        if user_input == "quit":
            break
        if user_input:
            ask_chain(rag_chain, user_input)


if __name__ == '__main__':
    main()
