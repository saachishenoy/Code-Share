
#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries for data processing, XML and HTML parsing, database operations, and text processing
import numpy as np
import pandas as pd
import os
import langchain  # Assuming a custom or third-party library for language processing
import lxml
from lxml import etree
from bs4 import BeautifulSoup
import sqlite3
import xml.etree.ElementTree as ET
from gensim.utils import simple_preprocess
from gensim.corpora.dictionary import Dictionary
from gensim.models import TfidfModel
from gensim import similarities

# Directory containing the data files to be processed
data = 'data/All_Publications_Part_1/'

# List all files in the specified data directory
input_files = os.listdir(data)

# Define a function to remove HTML tags from a given text string
def strip_html_tags(text):
    stripped = BeautifulSoup(text, 'html.parser').get_text().replace('
', ' ').replace('\', '').strip()
    return stripped

# Connect to SQLite database (or create it if it doesn't exist) and create a cursor object
conn = sqlite3.connect('subset_data.db')
cursor = conn.cursor()

# Create a new table in the database for storing the processed data, if it doesn't already exist
cursor.execute('''
    CREATE TABLE IF NOT EXISTS subset_table (
        goid INTEGER PRIMARY KEY,
        title TEXT,
        date TEXT,
        publication TEXT,
        text TEXT
    )
''')
conn.commit()

# Define a function to extract and insert data from an XML file into the SQLite database
def insert_data_from_xml(xml_file):
    tree = ET.parse(xml_file)  # Parse the XML file
    root = tree.getroot()  # Get the root element of the XML tree

    # Extract relevant information from the XML, handling missing elements gracefully
    goid = root.find('.//GOID').text if root.find('.//GOID') is not None else None
    title = root.find('.//Title').text if root.find('.//Title') is not None else None
    date = root.find('.//NumericDate').text if root.find('.//NumericDate') is not None else None
    publication = root.find('.//PublisherName').text if root.find('.//PublisherName') is not None else None

    # Attempt to find the text content in one of several possible tags, stripping HTML if found
    if root.find('.//FullText') is not None:
        text = root.find('.//FullText').text
    elif root.find('.//HiddenText') is not None:
        text = root.find('.//HiddenText').text
    elif root.find('.//Text') is not None:
        text = root.find('.//Text').text
    else:
        text = None
    if text is not None:
        text = strip_html_tags(text)

    # Insert the extracted information into the SQLite database
    cursor.execute('''
        INSERT INTO subset_table (goid, title, date, publication, text)
        VALUES (?, ?, ?, ?, ?)
    ''', (goid, title, date, publication, text))

# Iterate through each XML file in the specified directory, processing and inserting data into the database
xml_directory = data
for filename in os.listdir(xml_directory):
    if filename.endswith('.xml'):
        xml_file_path = os.path.join(xml_directory, filename)
        insert_data_from_xml(xml_file_path)

# Commit changes to the database and close the connection
conn.commit()
conn.close()
