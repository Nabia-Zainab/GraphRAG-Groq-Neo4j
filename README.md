# 🕸️ GraphRAG-Groq-Neo4j: Hybrid Retrieval System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Neo4j](https://img.shields.io/badge/Database-Neo4j-008CC1)
![Groq](https://img.shields.io/badge/LLM-Groq%20Llama%203.1-f55036)

## 🚀 Overview
**GraphRAG-Groq-Neo4j** is a cutting-edge Retrieval-Augmented Generation (RAG) application. Unlike traditional RAG, this system builds a structured **Knowledge Graph** from unstructured text using **Groq's Llama 3.1** and stores it in **Neo4j**. It uses a **Hybrid Retrieval** strategy (Vector + Graph) to answer complex queries.

## 🏗️ Tech Stack
* **LLM Engine:** Groq API (`llama-3.1-8b-instant`)
* **Graph Database:** Neo4j (AuraDB)
* **Vector Store:** ChromaDB (Local)
* **Orchestration:** LangChain
* **Frontend:** Streamlit

## 🛠️ Installation & Setup

**Step 1: Clone the Repository**
Run this command in your terminal:
`git clone https://github.com/YOUR_USERNAME/GraphRAG-Groq-Neo4j.git`
`cd GraphRAG-Groq-Neo4j`

**Step 2: Install Dependencies**
`pip install -r requirements.txt`

**Step 3: Configure Environment**
Create a file named `.env` and add your keys like this:
`GROQ_API_KEY=your_key_here`
`NEO4J_URI=your_uri_here`
`NEO4J_USERNAME=neo4j`
`NEO4J_PASSWORD=your_password`

**Step 4: Run the App**
`streamlit run app.py`

## 🧪 How It Works
1.  **Upload:** User uploads a PDF/TXT file.
2.  **Indexing:** The system extracts Entities & Relationships and saves them to Neo4j (Graph) and ChromaDB (Vector).
3.  **Querying:** When you ask a question, the system searches both the Graph and Vector store to give a precise answer.

---
*Built with ❤️ for the Open Source Community.*
