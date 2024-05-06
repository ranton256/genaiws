import os

from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# pip install -q langchain langchain-community langchain-core chromadb ollama PyPDF2 tiktoken

from PyPDF2 import PdfReader
from langchain.docstore.document import Document

# silence warnings from huggingface/tokenizers
os.environ['TOKENIZERS_PARALLELISM']="true"

MODEL_ID="gemma"


def get_document_text(file_path):
    docs = []
    title = os.path.basename(file_path)
    if file_path.lower().endswith('pdf'):
        pdf_reader = PdfReader(file_path)
        for num, page in enumerate(pdf_reader.pages):
            page = page.extract_text()
            doc = Document(page_content=page, metadata={'title': title, 'page': (num + 1)})
            docs.append(doc)

    else:
        # assume text if not pdf
        with open(file_path,"r") as f:
            doc_text = f.read()
        docs.append(doc_text)

    return docs


from langchain.text_splitter import RecursiveCharacterTextSplitter

# Download from:
# https://www.asge.org/docs/default-source/guidelines/asge-guideline-on-screening-and-surveillance-of-barrett-s-esophagus-2019-september-gie.pdf?sfvrsn=a562d352_2

file_path = 'acg_clinical_guideline__diagnosis_and_management.17.pdf'
the_text = get_document_text(file_path)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
chunks = text_splitter.split_documents(the_text)
print(f"Split into {len(chunks)} chunks")

llm = Ollama(model="gemma", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

oembed = OllamaEmbeddings(model="snowflake-arctic-embed")
vectorstore = Chroma.from_documents(documents=chunks, embedding=oembed)

retriever = vectorstore.as_retriever()

# These prompts are based on the example at
# https://python.langchain.com/docs/use_cases/summarization#option-2.-map-reduce

map_template = """The following is a set of documents
{docs}
Based on this list of docs, please identify the main themes
Helpful Answer:"""
map_prompt = PromptTemplate.from_template(map_template)
map_chain = LLMChain(llm=llm, prompt=map_prompt)

reduce_template = """The following is set of summaries:
{docs}
Take these and distill it into a final, consolidated summary of the main themes.
Helpful Answer:"""
reduce_prompt = PromptTemplate.from_template(reduce_template)

reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

combine_documents_chain = StuffDocumentsChain(
    llm_chain=reduce_chain, document_variable_name="docs"
)

reduce_documents_chain = ReduceDocumentsChain(
    combine_documents_chain=combine_documents_chain,
    # Used if documents exceed context for `StuffDocumentsChain`
    collapse_documents_chain=combine_documents_chain,
    # maximum number of tokens to group documents into.
    token_max=4000,
)

map_reduce_chain = MapReduceDocumentsChain(
    llm_chain=map_chain,
    reduce_documents_chain=reduce_documents_chain,
    document_variable_name="docs",
    return_intermediate_steps=False,
)

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, chunk_overlap=0
)

split_docs = text_splitter.split_documents(the_text)
summary_results = map_reduce_chain.run(split_docs)


print("*" * 10 + " FINISHED " + "*" * 10)
print("SUMMARY:\n", summary_results)


