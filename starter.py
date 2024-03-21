import os
from dotenv import load_dotenv

load_dotenv()

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

#load data and build an index
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)

#query your data
query_engine = index.as_query_engine()
response = query_engine.query("what did the author do growing up?")
print(response)

#view queries and events using logging
import logging
import sys

#You can set the level to DEBUG for verbose output, or use level=logging.INFO for less.
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

#sorting index locally
import os.path
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)

#check if storage already exists

PERSIST_DIR = "./storage"
if not os.path.exists(PERSIST_DIR):
    # load the documents and create the index
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

# Either way we can now query the index
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do in the year 2014?")
print(response)