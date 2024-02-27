#!/usr/bin/env python
# coding: utf-8

# In[18]:


import sqlite3
import numpy as np
import faiss
import openai
import torch
import transformers
from transformers import BertTokenizer, BertModel
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoModel, AutoTokenizer
from concurrent.futures import ProcessPoolExecutor, as_completed

# Function to fetch documents from the database based on their IDs
def fetch_documents_by_ids(db_path, document_ids):
    documents = []
    # Open a database connection
    with sqlite3.connect(db_path) as conn:
        for doc_id in document_ids:
            cursor = conn.cursor()
            # Execute query to fetch the text of the document by its ID
            cursor.execute("SELECT text FROM subset_table WHERE goid = ?", (doc_id,))
            result = cursor.fetchone()
            if result:
                documents.append(result[0])  # Append document text if found
            else:
                documents.append(None)  # Append None if document not found
    return documents

# Path to the SQLite database
db_path = 'subset_data.db'
# Fetch document IDs from the database (function call not shown in provided code)
document_ids = fetch_document_ids(db_path)

# Fetch the actual documents based on their IDs
documents = fetch_documents_by_ids(db_path, document_ids)

# Load a tokenizer and model from the transformers library
tokenizer = AutoTokenizer.from_pretrained("tokenizer")
model = AutoModel.from_pretrained("model")

# Load a pre-existing FAISS index
index_in = faiss.read_index("saved_index_subset_flat_512.index")

# Function to generate an embedding for a single piece of text
def generate_single_embedding(text, model, tokenizer):
    # Tokenize the text
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=256)
    # Move tensors to the same device as model
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
    
    with torch.no_grad():  # Disable gradient calculation
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:,0,:].cpu().numpy()  # Extract the embedding
    return embedding

# Function to search the FAISS index with a query
def search_faiss(query, faiss_index, k=5):
    faiss_index.nprobe = 10  # Set the number of probes for search
    encoded_query = generate_single_embedding(query, model, tokenizer)  # Generate embedding for the query
    distances, indices = faiss_index.search(encoded_query, k)  # Perform the search
    return [i for i in indices[0] if i < len(documents)]  # Return document indices

# Example search query
search_faiss("European headquarters", index_in)

