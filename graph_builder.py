import os
from typing import List, Optional
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_community.graphs import Neo4jGraph
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv(override=True)

# Define Data Models for Extraction
class Node(BaseModel):
    id: str = Field(description="Unique identifier for the entity")
    type: str = Field(description="Type of the entity (e.g., Person, Organization, Concept)")

class Relationship(BaseModel):
    source: str = Field(description="Source entity ID")
    target: str = Field(description="Target entity ID")
    type: str = Field(description="Type of relationship (e.g., WORKS_FOR, LOCATED_IN)")
    description: Optional[str] = Field(description="Brief description of the relationship context")

class GraphData(BaseModel):
    nodes: List[Node]
    relationships: List[Relationship]

class GraphBuilder:
    def __init__(self):
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        self.graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD")
        )

    def extract_graph_data(self, text: str) -> GraphData:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a knowledge graph extractor. Extract entities (nodes) and relationships from the text.
            Focus on key concepts, people, organizations, and their interactions.
            Return the result in a structured JSON format matching the schema."""),
            ("human", "Text: {text}")
        ])
        
        chain = prompt | self.llm.with_structured_output(GraphData)
        return chain.invoke({"text": text})

    def ingest_documents(self, documents: List[Document]):
        print(f"Processing {len(documents)} documents...")
        
        # Clear existing graph (Optional: make this configurable)
        # self.graph.query("MATCH (n) DETACH DELETE n")
        
        for doc in documents:
            # Extract graph data
            try:
                data = self.extract_graph_data(doc.page_content)
                
                # Create Nodes
                for node in data.nodes:
                    self.graph.query(
                        "MERGE (n:Entity {id: $id}) SET n.type = $type",
                        {"id": node.id, "type": node.type}
                    )
                    # Add specific label if possible, but dynamic labels are tricky in parameters
                    # We use a generic 'Entity' label and a 'type' property for simplicity
                
                # Create Relationships
                for rel in data.relationships:
                    cypher = f"""
                    MATCH (s:Entity {{id: $source}})
                    MATCH (t:Entity {{id: $target}})
                    MERGE (s)-[r:{rel.type.upper().replace(' ', '_')}]->(t)
                    SET r.description = $description
                    """
                    self.graph.query(
                        cypher,
                        {
                            "source": rel.source, 
                            "target": rel.target, 
                            "description": rel.description
                        }
                    )
                print(f"Processed document chunk: {doc.page_content[:50]}...")
            except Exception as e:
                print(f"Error processing chunk: {e}")

        print("Graph ingestion complete.")
