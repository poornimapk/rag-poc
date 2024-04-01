from dotenv import load_dotenv
import json
from llama_index.core import (
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    VectorStoreIndex,
)
import pytest
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.evaluation import (
    RelevancyEvaluator,
)
from llama_index.llms.openai import OpenAI


def load_json_file(file_name):
    with open(file_name, "r") as file:
        json_data = json.load(file)
    print("Type of JSON Object: ", type(json_data))
    return json_data

def generate_input_output_pairs_from_json(json_data: dict):
    in_out_pairs = []
    queries: dict = json_data.get("queries")
    responses: dict = json_data.get("responses")
    for key, value in queries.items():
        in_out_pairs.append({"input": value, "expected_output": responses.get(key)})
    # print(in_out_pairs)
    return in_out_pairs


load_dotenv()

#Initialize SimpleDirectoryReader with the path to your directory containing documents
documents = SimpleDirectoryReader("./data/oracle/").load_data()

# Initialize RelevancyEvaluator
llm3 = OpenAI(model="gpt-3.5-turbo")
evaluator_gpt3 = RelevancyEvaluator(llm=llm3)

# Initialize RelevancyEvaluator
llm4 = OpenAI(model="gpt-4")
evaluator_gpt4 = RelevancyEvaluator(llm=llm4)

Settings.chunk_size = 512
Settings.chunk_overlap = 50

# Initialize query engine
vector_store = MilvusVectorStore(dim=1536, overwrite=True)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context,# service_context=ServiceContext.from_defaults(chunk_size=512)
)

query_engine = index.as_query_engine(similarity_top_k=4)

# Load JSON file
json_data = load_json_file("./testdata/testdata.json")

input_output_pairs = generate_input_output_pairs_from_json(json_data)

@pytest.mark.parametrize(
        "input_output_pair",
        input_output_pairs
)
def test_llamaindex(input_output_pair: dict):
    query = input_output_pair.get("input", None)
    expected_output = input_output_pair.get("expected_output", None)

    actual_output = query_engine.query(query)

    eval_result = evaluator_gpt4.evaluate_response(
        query=query,
        response=actual_output,
    )

    assert eval_result.passing == True
