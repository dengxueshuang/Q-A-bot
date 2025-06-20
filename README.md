# Markdown QA with LangChain + Ollama

This is a local knowledge base Q&A system based on LangChain and Ollama, used to build a vector index from Markdown documents and perform natural language Q&A. It supports Chinese/English mixed input and is compatible with lightweight local models such as deepseek-rl-1.5b.

## Features
Automatically loads .md files from the specified directory

Recursively splits documents into chunks using a text splitter

Builds a local vector database using FAISS

Uses SentenceTransformer as the embedding model

Integrates local large language models via Ollama for answering questions

Runs completely locally, ensuring privacy and security


### Installation & Running
### Create a virtual environment:
python -m venv .venv
.\.venv\Scripts\activate  
### Install dependencies:
pip install -r requirements.txt
### Prepare Ollama Model
Make sure you’ve installed and are running Ollama. For example, use the following command to pull a model:
ollama pull deepseek-r1:1.5b
### Run:
python main.py
