import os
import time
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex,StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv

def query_documents():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    load_dotenv()
    start = time.time()
    
    # Load documents
    document_reader = SimpleDirectoryReader("./data/oracle/")
    documents = document_reader.load_data()
    
    # Set up vector store
    vector_store = MilvusVectorStore(dim=1536, overwrite=True)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Create index
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    
    # Set up language model
    llm = OpenAI(model="gpt-3.5-turbo")
    
    # Create query engine
    query_engine = index.as_query_engine(llm=llm)
    
    # Perform query
    #query = "What are Notes Payable and Other borrowings can you explain me in detail ? How much amount we have borrowed under a delayed draw term loan Credit agreement ? "
    #query = "Can you mention the number on how much we are going to spend on Total cloud services and license support revenues ?"
    query = "What is the cloud and license revenue for the year 2023?"
    #query1 = "Total Number of Shares Purchased as Part of Publicly Announced Program from February 1, 2024—February 29, 2024 " --> Table information.
    response = query_engine.query(query)
    print(response)
    
    end = time.time()
    print("Execution time:", (end - start), "seconds")
    #print("Execution time:", (end - start) / 60, "minutes")
    #print("Execution time: ", (end-start) * 10**3, "ms")

# Call the function to execute the code
query_documents()