#view queries and events using logging
import time
import textwrap
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import Settings

def main():
    #You can set the level to DEBUG for verbose output, or use level=logging.INFO for less.
    #logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    #logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
    load_dotenv()

    start = time.time()

    # Initialize SimpleDirectoryReader with the path to your directory containing documents
    documents = SimpleDirectoryReader("./data/oracle/").load_data()

    Settings.chunk_size = 512
    Settings.chunk_overlap = 50
    
    vector_store = MilvusVectorStore(dim=1536, overwrite=True)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context
    )

    query_engine = index.as_query_engine(similarity_top_k=4)
    #response = query_engine.query("What is the total revenues in Asia Pacific region in millions in the year 2020?")
    #print(textwrap.fill(str(response), 100))

    #response = query_engine.query("What is the trend in earnings per share from 2020 to 2022?")
    #print(textwrap.fill(str(response), 100))

    #response = query_engine.query("If there is an increase, what is the percentage of increase?")
    #print(textwrap.fill(str(response), 100))

    #response = query_engine.query("If there is a decrease, what is the percentage of decrease?")
    #print(textwrap.fill(str(response), 100))

    response = query_engine.query("What is the cloud and license revenue in millions for the year 2023?")
    print(response)

    response = query_engine.query("What is the total revenue in millions for the year 2023?")
    print(response)

    response = query_engine.query("Who holds the majority of the non-marketable investments as of February 29 2024?")
    print(response)

    end = time.time()
    
    print("Execution time: ", (end-start) * 10**3, "ms")

if __name__ == '__main__':
    main()