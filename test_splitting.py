import os
import time
from llama_index.core import SimpleDirectoryReader
from dotenv import load_dotenv
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex,StorageContext
from pymilvus import MilvusClient
from llama_index.core import Settings

def query_documents():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    load_dotenv()
    start = time.time()
    
    #Settings.text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
    # Connect to Milvus (replace number with actual port)
    #client = MilvusClient(uri="Place your URL here")

    # Load documents
    documents = SimpleDirectoryReader("./data/oracle/").load_data()
    # Create or use a vector store (optional)
    vector_store = MilvusVectorStore(dim=1536, collection_name="test1", overwrite=True)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Split documents into nodes (sentences)
    text_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = text_parser.get_nodes_from_documents(documents)
    #print(nodes[:5]) #:- Checking whether the text are chunked into nodes.

    # Add nodes to the vector store
    #vector_store.add(nodes)

    # Create index
    '''index = VectorStoreIndex.from_documents(
        documents, 
        storage_context=storage_context,
        transformations=[SentenceSplitter(chunk_size=512, chunk_overlap=50)],
        )'''
    index = VectorStoreIndex(nodes=nodes, storage_context=storage_context)
    
    # Set up language model
    llm = OpenAI(model="gpt-3.5-turbo")
    
    # Create query engine
    query_engine = index.as_query_engine(llm=llm)
    
    # Perform query
    query1 ="How many full-time employees were reported by Oracle Corporation as of May 31, 2022?"
    #query1 ="What is the net income of Oracle corporation by February 28 2023?"
    #query1 = "Which company acquisition was completed by Oracle Corporation on June 8, 2022?"
    response = query_engine.query(query1)
    print(response)

query_documents()