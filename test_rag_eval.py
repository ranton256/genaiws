# Use DeepEval for evaluating our RAG result set.

# pip install -U deepeval toml
# pip install -U pytest

from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric
)

from deepeval.test_case import LLMTestCase
from deepeval import evaluate

import deepeval
from deepeval import assert_test

import pytest
import pickle
import toml
import os


# Setup OpenAI API which it uses by default
with open('secrets.toml', 'r') as f:
    config = toml.load(f)
    OPENAI_API_KEY = config['OPENAI_API_KEY']
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

contextual_precision = ContextualPrecisionMetric()
contextual_recall = ContextualRecallMetric()
contextual_relevancy = ContextualRelevancyMetric()

answer_relevancy = AnswerRelevancyMetric()
faithfulness = FaithfulnessMetric()

# load our test data

# Columns are: question,answer,contexts,ground_truth
dataset = EvaluationDataset()
dataset.add_test_cases_from_csv_file(
    file_path="test_data_for_eval.csv",
    input_col_name="question",
    actual_output_col_name="answer",
    expected_output_col_name="ground_truth",
    retrieval_context_col_name="cleaned_context",
    retrieval_context_col_delimiter= ";"
)

@pytest.mark.parametrize(
    "test_case",
    dataset,
)
def test_rag_retrieval(test_case: LLMTestCase):
    contextual_precision = ContextualPrecisionMetric(threshold=0.6)
    contextual_recall = ContextualRecallMetric(threshold=0.3)
    contextual_relevancy = ContextualRelevancyMetric(threshold=0.5)
    
    assert_test(test_case, [contextual_precision, contextual_recall, contextual_relevancy])


@pytest.mark.parametrize(
    "test_case",
    dataset,
)
def test_rag_generation(test_case: LLMTestCase):
    answer_relevancy = AnswerRelevancyMetric(threshold=0.5)
    faithfulness = FaithfulnessMetric(threshold=0.5)

    assert_test(test_case, [answer_relevancy, faithfulness])


@deepeval.on_test_run_end
def function_to_be_called_after_test_run():
    print("Test finished!")
    