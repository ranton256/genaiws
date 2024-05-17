import os

import pandas as pd
import sys

import toml
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.embeddings import OllamaEmbeddings, HuggingFaceInferenceAPIEmbeddings
from langchain_community.llms.ollama import Ollama

from rag import get_model, setup_rag, setup_retriever
from datasets import Dataset

def run_batch(queries):
    responses = []
    context_docs = []

    def save_context(docs):
        # print(f"Saving context of {len(docs)} docs")
        context_docs.append([doc.page_content for doc in docs])
        return docs

    model = get_model()
    retriever, vs = setup_retriever()
    rag_chain = setup_rag(model, retriever, process_docs=save_context)

    # This could use batch inference.
    n_total = len(queries)
    for i, query in enumerate(queries):
        print(f"Processing query #{i+1} of {n_total}")
        response = rag_chain.invoke(query)
        responses.append(response)
    print("Batch done.")

    df = pd.DataFrame({"question": queries, "answer": responses, "contexts": context_docs})
    return df


def main():
    testset_path = "testset.csv"
    results_path = "rag_results.csv"

    testset_df = pd.read_csv(testset_path)

    # create smaller subset for rapid testing.
    # testset_df = testset_df.head(3)
    
    queries = testset_df['Question']

    df = run_batch(queries)
    df['ground_truth'] = testset_df['Correct Answer']

    # escape ';' then use it as join delimiter.
    df["cleaned_contexts"] = df["contexts"].apply(lambda items: ';'.join([s.replace(';', ' ') for s in items]))
    df["contexts"] = df["cleaned_contexts"] 
    
    df.to_csv(results_path)

if __name__ == '__main__':
    main()
