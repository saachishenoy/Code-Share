#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Import necessary libraries
import sqlite3  # For SQLite database operations
import numpy as np  # For numerical operations and array handling
import openai  # OpenAI's API for accessing AI models
import torch  # PyTorch, a library for tensor computations and neural networks
import transformers  # Library providing interfaces and pre-trained models for natural language processing
from transformers import BertTokenizer, BertModel  # Specific imports for BERT tokenizer and model
from concurrent.futures import ThreadPoolExecutor  # For executing calls asynchronously
from transformers import AutoModel, AutoTokenizer  # For loading models and tokenizers in a flexible way
from concurrent.futures import ProcessPoolExecutor, as_completed  # For parallel execution of calls

# Function to fetch document IDs from a SQLite database
def fetch_document_ids(db_path):
    # Establish a connection to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # Execute a query to select all document IDs
    cursor.execute("SELECT id FROM documents")  # Adjust the query based on your table schema
    # Retrieve all IDs from the query result
    ids = [row[0] for row in cursor.fetchall()]
    # Close the database connection
    conn.close()
    # Return the list of document IDs
    return ids

# Function to fetch document texts by their IDs
def fetch_documents_by_ids(db_path, document_ids):
    documents = []
    # Open a single connection to the database
    with sqlite3.connect(db_path) as conn:
        for doc_id in document_ids:
            cursor = conn.cursor()
            # Execute a query to select text of a document by ID
            cursor.execute("SELECT text FROM subset_table WHERE goid = ?", (doc_id,))
            result = cursor.fetchone()
            # If the document is found, append its text to the list, otherwise append None
            if result:
                documents.append(result[0])
            else:
                documents.append(None)
    # Return the list of document texts
    return documents

# Main script execution starts here

# Define the path to the SQLite database
db_path = 'subset_data.db'
# Fetch document IDs from the database
document_ids = fetch_document_ids(db_path)
# Fetch the documents' texts by their IDs
documents = fetch_documents_by_ids(db_path, document_ids)

# Initialize the tokenizer and model for embedding generation
tokenizer = AutoTokenizer.from_pretrained('tokenizer')
model = AutoModel.from_pretrained('model')

# Set the device to GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
# Move the model to the selected device
model = model.to(device)

# Function to generate embeddings for a list of texts
def generate_embeddings(texts, model, tokenizer, batch_size=256):
    # Put the model in evaluation mode
    model.eval()
    embeddings = []
    # Process texts in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        # Tokenize the batch of texts
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512)
        # Move inputs to the selected device
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
        # Perform inference without calculating gradients
        with torch.no_grad():
            outputs = model(**inputs)
            # Extract embeddings and move them back to CPU
            embeddings.extend(outputs.last_hidden_state[:,0,:].cpu().numpy())
    # Return the list of embeddings
    return embeddings

# Measure the time taken to generate embeddings
import time
start_time = time.time()
# Generate embeddings for the documents
embeddings = generate_embeddings(documents, model, tokenizer)
end_time = time.time()
duration = end_time - start_time
print(f"Embedding generation took {duration:.2f} seconds for {len(documents)} samples.")

# Save the generated embeddings to a file
np.save('embeddings_docs_bert_base.npy', embeddings)
