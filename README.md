# 🕸️ GraphRAG-Groq-Neo4j: Hybrid Retrieval System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Neo4j](https://img.shields.io/badge/Database-Neo4j-008CC1)
![Groq](https://img.shields.io/badge/LLM-Groq%20Llama%203.1-f55036)
![LangChain](https://img.shields.io/badge/Orchestration-LangChain-white)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B)

## 🚀 Overview
**GraphRAG-Groq-Neo4j** is a cutting-edge Retrieval-Augmented Generation (RAG) application that leverages the power of **Knowledge Graphs** combined with **Vector Search**.

Unlike traditional RAG which relies solely on vector similarity, this system builds a structured Knowledge Graph from unstructured text (PDFs/TXT) using **Groq's Llama 3.1** and stores it in **Neo4j**. It performs a **Hybrid Retrieval** strategy to answer complex queries that require multi-hop reasoning and relationship understanding.

## 🏗️ Architecture & Tech Stack

This project implements a "Graph + Vector" hybrid approach:

* **LLM Engine:** Groq API (running `llama-3.1-8b-instant`) for ultra-fast inference.
* **Graph Database:** Neo4j (AuraDB) for storing entities and relationships.
* **Vector Store:** ChromaDB (Local) for semantic similarity search.
* **Embeddings:** HuggingFace `all-MiniLM-L6-v2` (Local & Open Source).
* **Orchestration:** LangChain for managing the RAG pipeline.
* **Frontend:** Streamlit for an interactive user interface.

## ✨ Key Features

* **📄 Document Ingestion:** Upload PDF or TXT files directly via the UI.
* **🧠 Automated Graph Construction:** Extracts Entities and Relationships automatically using Llama 3.1.
* **🔍 Hybrid Search:** Combines results from:
    1.  **Vector Search:** Finds semantically similar text chunks.
    2.  **Graph Traversal:** Retrieves connected concepts and hidden relationships.
* **⚡ High Performance:** Uses Groq for near real-time processing and response generation.
* **💬 Interactive Chat:** Context-aware Q&A interface.

## 🛠️ Installation & Setup

### Prerequisites
* Python 3.8+
* Neo4j Database (AuraDB Free Tier recommended)
* Groq API Key

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/GraphRAG-Groq-Neo4j.git](https://github.com/YOUR_USERNAME/GraphRAG-Groq-Neo4j.git)
cd GraphRAG-Groq-Neo4j

2. Install Dependencies
Bash

pip install -r requirements.txt
3. Environment Configuration
Create a .env file in the root directory and add your credentials:

Ini, TOML

GROQ_API_KEY=your_groq_api_key_here
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password
4. Run the Application
Bash

streamlit run app.py
📂 Project Structure
graphrag-app/
├── app.py               # Main Streamlit Frontend Application
├── backend/
│   ├── rag_chain.py     # LangChain Logic (Hybrid Retriever)
│   ├── graph_builder.py # Text-to-Graph Extraction Logic
│   └── config.py        # Environment Configuration
├── requirements.txt     # Python Dependencies
├── .env                 # API Keys (Not tracked in Git)
└── README.md            # Documentation
🧪 How It Works
Upload: User uploads a document. The text is chunked.

Extraction: The LLM identifies nodes (Entities) and edges (Relationships) from the chunks.

Indexing:

Text chunks are embedded and saved to ChromaDB.

Entities and relationships are pushed to Neo4j via Cypher queries.

Querying:

The user asks a question.

The system performs a Vector Search to get relevant context.

Simultaneously, it queries the Graph to find related entities (neighborhood retrieval).

Contexts are combined and passed to Llama 3.1 to generate a precise answer.
