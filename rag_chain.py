import os
from typing import List
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_neo4j import Neo4jGraph
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv(override=True)

class GraphRAGChain:
    def __init__(self):
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        
        # Initialize Embeddings (Local, Free)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Initialize Vector Store (Persistent)
        self.vector_store = Chroma(
            persist_directory="./chroma_db",
            embedding_function=self.embeddings
        )
        
        # Initialize Graph Store
        self.graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD")
        )

    def add_documents_to_vector_store(self, documents: List[Document]):
        self.vector_store.add_documents(documents)
        
    def get_graph_context(self, query: str) -> str:
        # Simple approach: Extract entities from query and find neighbors
        # 1. Extract entities from query using LLM
        prompt = ChatPromptTemplate.from_template(
            "Extract the main entities (people, organizations, concepts) from this query as a comma-separated list: {query}"
        )
        chain = prompt | self.llm | StrOutputParser()
        entities_str = chain.invoke({"query": query})
        entities = [e.strip() for e in entities_str.split(",")]
        
        context = []
        for entity in entities:
            # Cypher to find node and 1-hop neighbors
            cypher = f"""
            MATCH (n:Entity) WHERE n.id CONTAINS $entity
            OPTIONAL MATCH (n)-[r]-(m)
            RETURN n.id, type(r), m.id, r.description
            LIMIT 10
            """
            try:
                result = self.graph.query(cypher, {"entity": entity})
                for record in result:
                    # Format: "Entity A -[RELATION]-> Entity B (Description)"
                    if record['r.description']:
                        context.append(f"{record['n.id']} {record['type(r)']} {record['m.id']} ({record['r.description']})")
                    else:
                        context.append(f"{record['n.id']} {record['type(r)']} {record['m.id']}")
            except Exception:
                continue
                
        return "\n".join(context)

    def get_chain(self):
        # Vector Retriever
        vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        
        def hybrid_retrieval(query):
            # 1. Vector Search
            vector_docs = vector_retriever.invoke(query)
            vector_context = "\n".join([d.page_content for d in vector_docs])
            
            # 2. Graph Search
            graph_context = self.get_graph_context(query)
            
            return f"""
            Vector Context:
            {vector_context}
            
            Graph Context:
            {graph_context}
            """
            
        template = """Answer the question based only on the following context:
        {context}
        
        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        
        chain = (
            {"context": hybrid_retrieval, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return chain
