# Use DeepEval for evaluating our RAG result set.

import deepeval
from deepeval import assert_test
from deepeval.dataset import EvaluationDataset
from deepeval.test_case import LLMTestCase
from deepeval import evaluate
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    AnswerRelevancyMetric, 
    FaithfulnessMetric
)

import pytest
import toml
import os


# runs on subset of test cases for speedy debugging
want_subset = False

# Setup OpenAI API key
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
    file_path="rag_results.csv",
    input_col_name="question",
    actual_output_col_name="answer",
    expected_output_col_name="ground_truth",
    retrieval_context_col_name="contexts",
    retrieval_context_col_delimiter= ";"
)

if want_subset:
    subset_cases = list(dataset.test_cases)[:3]
    dataset.test_cases = subset_cases

@pytest.mark.parametrize(
    "test_case",
    dataset.test_cases,
)
def test_rag_retrieval(test_case: LLMTestCase):
    contextual_precision = ContextualPrecisionMetric(threshold=0.3)
    contextual_recall = ContextualRecallMetric(threshold=0.3)
    contextual_relevancy = ContextualRelevancyMetric(threshold=0.3)
    
    assert_test(test_case, [contextual_precision, contextual_recall, contextual_relevancy])


@pytest.mark.parametrize(
    "test_case",
    dataset.test_cases,
)
def test_rag_generation(test_case: LLMTestCase):
    answer_relevancy = AnswerRelevancyMetric(threshold=0.5)
    faithfulness = FaithfulnessMetric(threshold=0.5)

    assert_test(test_case, [answer_relevancy, faithfulness])


@deepeval.on_test_run_end
def function_to_be_called_after_test_run():
    print("Test finished!")
    