#!/usr/bin/env python
# coding: utf-8

# In[1]:
# Import necessary libraries
import sqlite3  # For SQLite database operations
import numpy as np  # For numerical operations and handling arrays
import faiss  # Efficient similarity search and clustering of dense vectors
import openai  # OpenAI's API for accessing AI models
import torch  # PyTorch, a library for tensor computations and neural networks
import transformers  # Library for natural language processing
from transformers import BertTokenizer, BertModel  # BERT model and tokenizer for text processing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed  # For parallel execution

# Define a function to create a quantized FAISS index from document embeddings
def create_quantized_faiss_index(encoded_docs):
    # Get the dimensionality of the embeddings
    dimension = encoded_docs.shape[1]
    # Determine the number of partitions (nlist) for the index; capped at 2048
    nlist = min(len(encoded_docs), 2048)
    # Create a quantizer using L2 distance
    quantizer = faiss.IndexFlatL2(dimension)
    # Initialize the index with specified parameters for vector quantization
    index = faiss.IndexIVFPQ(quantizer, dimension, nlist, 256, 8)
    # Set the number of probes to 20 for searching
    index.nprobe = 20
    # Ensure the index is not yet trained
    assert not index.is_trained
    # Train the index with the provided embeddings
    index.train(encoded_docs)
    # Add the embeddings to the index
    index.add(encoded_docs)
    # Return the created and populated index
    return index

# Load embeddings from a file
embeddings_loaded = np.load('embeddings_docs_bert_base.npy', allow_pickle=True)

# Create a quantized FAISS index using the loaded embeddings
index = create_quantized_faiss_index(embeddings_loaded)

# Save the created FAISS index to a file for future use
faiss.write_index(index, "saved_index_subset_flat_512.index")




