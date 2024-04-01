from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.core import (
    SimpleDirectoryReader,
)
from llama_index.core.evaluation import (
    DatasetGenerator,
)


def main():
    load_dotenv()

    #set context for llm provider
    # gpt_4_context = ServiceContext.from_defaults(
    #     llm=OpenAI(model="gpt-4", temperature=0.3)
    # )
    llm = OpenAI(model="gpt-3.5-turbo")
    # Initialize SimpleDirectoryReader with the path to your directory containing documents
    documents = SimpleDirectoryReader("./data/oracle/").load_data()

    #instantiate a DatasetGenerator
    dataset_generator = DatasetGenerator.from_documents(
        documents,
        llm=llm,
        num_questions_per_chunk=2,
        show_progress=True
    )
    rag_dataset = dataset_generator.generate_dataset_from_nodes(10)

    rag_dataset.save_json("./testdata/testdata.json")
    print("Done")

    # RagDatasetGenerator do not have a provision to 
    # set the number of dataset to generate and it throws 
    # Exception since a huge data is getting generated
    # This is very recently released and is not stable.
    # But the steps are below.

    # dataset_generator = RagDatasetGenerator.from_documents(
    #     documents=documents,
    #     llm=llm,
    #     num_questions_per_chunk=2,
    # )

    # rag_dataset = dataset_generator.generate_dataset_from_nodes()
    # rag_dataset.save_json("./testdata/testdata.json")

if __name__ == '__main__':
    main()