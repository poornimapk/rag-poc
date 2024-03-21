#view queries and events using logging
import logging
import sys
import os
import nest_asyncio
import time
import textwrap
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore

def main():
    #You can set the level to DEBUG for verbose output, or use level=logging.INFO for less.
    #logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    #logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
    load_dotenv()

    start = time.time()

    #use SimpleDirectoryReader to parse our file
    reader = SimpleDirectoryReader("./data/oracle/")
    documents = reader.load_data()

    print(documents[0].doc_id)

    vector_store = MilvusVectorStore(dim=1536, overwrite=True)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context
    )

    query_engine = index.as_query_engine()
    response = query_engine.query("What is the main business of Oracle Corporation?")
    print(textwrap.fill(str(response), 100))
    #response = query_engine.query("How much shares or other units to be sold?")
    #print(textwrap.fill(str(response), 100))
    
    end = time.time()
    
    print("Execution time: ", (end-start) * 10**3, "ms")

if __name__ == '__main__':
    main()