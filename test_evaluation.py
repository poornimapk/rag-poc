#view queries and events using logging
import time
import textwrap
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex, StorageContext, ServiceContext
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.core.llama_dataset import (
    LabeledRagDataset,
    LabeledPairwiseEvaluatorDataset,
    CreatedBy,
    CreatedByType,
    LabeledRagDataExample,
)
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from llama_index.core.llama_dataset.generator import LabelledRagDataset
from llama_index.core.evaluation import (
    DatasetGenerator,
    CorrectnessEvaluator,
    FaithfulnessEvaluator,
    RelevancyEvaluator
) 

def main():
    load_dotenv()

    start = time.time()

    llm_gpt3 = OpenAI("gpt-3.5-turbo")
    llm_gpt4 = OpenAI("gpt-4")
    
    #evaluator_gpt3 = RelevancyEvaluator(llm_gpt3)
    evaluator_gpt4 = RelevancyEvaluator(llm_gpt4)

    # Initialize SimpleDirectoryReader with the path to your directory containing documents
    documents = SimpleDirectoryReader("./data/oracle/").load_data()

    # create dataset_generator for generating eval questions
    
    dataset_generator = RagDatasetGenerator.from_documents(
        documents=documents,
        #llm=llm_gpt4,
        #num_questions_per_chunk=1,
        )

    # create eval_questions from dataset_generator
    #eval_questions = dataset_generator.generate_questions_from_nodes(3)
    eval_questions = dataset_generator.generate_dataset_from_nodes(3)

    print(eval_questions[:10])
    print("Total number of queries: ", len(eval_questions))

    
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50

    vector_store = MilvusVectorStore(dim=1536, overwrite=True)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context,# service_context=ServiceContext.from_defaults(chunk_size=512)
    )

    query_engine = index.as_query_engine(similarity_top_k=4)
    #query = "What is the total revenues of Oracle corporation by February 29 2024?"
    #query = "Who holds the non marketable investments as of February 29, 2024?"
    #query = "What is the net income of Oracle corporation by February 28 2023?"
    #query = "Which company acquisition was completed by Oracle Corporation on June 8, 2022?"
    print("Query is: ", eval_questions[45])
    response_from_vectordb = query_engine.query(eval_questions[45])
    #print("Response is: ", textwrap.fill(str(response), 100))
    print("Response is: ", response_from_vectordb)

    eval_result = evaluator_gpt4.evaluate_response(
        query=eval_questions[45],
        response=response_from_vectordb,
    )

    #print(eval_result)
    print("Evaluation result: ", eval_result.passing, " Reasoning: ", eval_result.feedback, "Eval Response: ", eval_result.response, "Eval Score: ", eval_result.score)
    #reference = "The total revenues of Oracle Corporation by February 29 2024 is $13,280 millions."
    #reference = "The net income of Oracle corporation as of February 28, 2023 amounted to 5,184."
    '''
    reference = "On June 8, 2022, Oracle completed our acquision of Cerner Corporaon (Cerner), a provider of digital informaon systems used within hospitals and health systems that are designed to enable medical professionals to deliver beer healthcare to individual paents and communies."
    evaluator = CorrectnessEvaluator(llm=llm)

    result = evaluator.evaluate(
        query=query,
        response=str(response),
        reference=reference,
    )
    print(result)
    print(result.feedback)
    #eval_questions = [
    #    "What is the total revenues of Oracle corporation by February 29 2024?",
    #    "What is the net income of Oracle corporation by February 28 2023?",
    #    "Who holds the non marketable investments as of February 29, 2024?",
    #    "Which company acquisition was completed by Oracle Corporation on June 8, 2022?",
    #    "What are the various intangible assets and goodwill of Oracle corporation as of May 31, 2023?"
    #]

    #eval_answers = [
    #    "15,940",  # incorrect answer 13,280 - correct answer
    #    # incorrect answer - "7,323",  correct answer - "5,184"
    #    "The majority of the non-marketable investments held as of these dates were with Ampere Computing Holdings LLC (Ampere), a related party entity in which we have an ownership interest of approximately 29% as of February 29, 2024",
    #    "On June 8, 2022, Oracle completed our acquision of Cerner Corporaon (Cerner), a provider of digital informaon systems used within hospitals and health systems that are designed to enable medical professionals to deliver beer healthcare to individual paents and communies.",
    #    "The intangible assets are Developed technology, Cloud services and license support agreements and related relationships, Cloud license and on-premise license agreements and related relaonships",
    #]

    #eval_answers = [[a] for a in eval_answers]

    

    '''
    end = time.time()
    
    print("Execution time: ", (end-start) * 10**3, "ms")

if __name__ == '__main__':
    main()