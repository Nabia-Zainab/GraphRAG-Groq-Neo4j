import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from graph_builder import GraphBuilder
from rag_chain import GraphRAGChain
from streamlit_agraph import agraph, Node, Edge, Config
from dotenv import load_dotenv

load_dotenv(override=True)

st.set_page_config(page_title="GraphRAG with Groq & Neo4j", layout="wide")

st.title("🕸️ GraphRAG: Knowledge Graph Enhanced GenAI")
st.markdown("Powered by **Groq (Llama 3.1)** and **Neo4j**")

# Tabs
tab1, tab2 = st.tabs(["💬 Chat", "🕸️ Graph Visualization"])

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = GraphRAGChain()

if "graph_builder" not in st.session_state:
    st.session_state.graph_builder = GraphBuilder()

# Sidebar for File Upload
with st.sidebar:
    st.header("📄 Document Upload")
    uploaded_files = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"], accept_multiple_files=True)
    
    if st.button("Process Documents") and uploaded_files:
        with st.spinner("Processing Documents..."):
            for uploaded_file in uploaded_files:
                st.write(f"Processing: {uploaded_file.name}")
                # Save temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                # Load Document
                if uploaded_file.name.endswith(".pdf"):
                    loader = PyPDFLoader(tmp_path)
                else:
                    loader = TextLoader(tmp_path)
                
                docs = loader.load()
                
                # Split Text
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents(docs)
                
                # Ingest into Graph
                st.session_state.graph_builder.ingest_documents(splits)
                
                # Add to Vector Store
                st.session_state.rag_chain.add_documents_to_vector_store(splits)
                
                os.remove(tmp_path)
            
            st.success("All Documents Processed! You can now chat with your data.")

# Chat Interface
with tab1:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking (Traversing Graph + Searching Vectors)..."):
                try:
                    chain = st.session_state.rag_chain.get_chain()
                    response = chain.invoke(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"An error occurred: {e}")

# Graph Visualization
with tab2:
    st.header("Knowledge Graph Visualization")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        focus_node = st.text_input("Focus on Entity (Leave empty for overview)", placeholder="e.g., 'Artificial Intelligence'")
    with col2:
        st.write("") # Spacer
        st.write("")
        refresh_btn = st.button("Refresh Graph")

    if refresh_btn or focus_node:
        try:
            # Query Neo4j
            driver = st.session_state.graph_builder.graph._driver
            with driver.session() as session:
                if focus_node:
                    # Show node and its neighbors
                    query = f"""
                    MATCH (n)-[r]-(m) 
                    WHERE n.id CONTAINS '{focus_node}' 
                    RETURN n.id AS source, type(r) AS type, m.id AS target 
                    LIMIT 50
                    """
                    st.caption(f"Showing neighborhood of: **{focus_node}**")
                else:
                    # Show most connected nodes (Overview)
                    query = """
                    MATCH (n)-[r]->(m) 
                    WITH n, count(r) as rel_count 
                    ORDER BY rel_count DESC 
                    LIMIT 10 
                    MATCH (n)-[r]->(m) 
                    RETURN n.id AS source, type(r) AS type, m.id AS target
                    """
                    st.caption("Showing top connected clusters (Overview)")

                result = session.run(query)
                
                nodes = set()
                edges = []
                
                for record in result:
                    source = record["source"]
                    target = record["target"]
                    rel_type = record["type"]
                    
                    nodes.add(source)
                    nodes.add(target)
                    edges.append(Edge(source=source, label=rel_type, target=target, type="CURVE_SMOOTH"))
                
                if not nodes:
                    st.warning("No nodes found. Try a different search term or process some documents.")
                else:
                    # Create agraph Nodes
                    agraph_nodes = [Node(id=n, label=n, size=25, shape="circularImage", image="https://cdn-icons-png.flaticon.com/512/3135/3135715.png") for n in nodes]
                    
                    config = Config(width=900, 
                                    height=700, 
                                    directed=True, 
                                    nodeHighlightBehavior=True, 
                                    highlightColor="#F7A7A6", 
                                    collapsible=True,
                                    physics={
                                        "enabled": True,
                                        "stabilization": {
                                            "enabled": True,
                                            "iterations": 200
                                        },
                                        "barnesHut": {
                                            "gravitationalConstant": -2000,
                                            "centralGravity": 0.1,
                                            "springLength": 250,
                                            "springConstant": 0.01,
                                            "damping": 0.09,
                                            "avoidOverlap": 1
                                        }
                                    })
                    
                    return_value = agraph(nodes=agraph_nodes, edges=edges, config=config)
        except Exception as e:
            st.error(f"Could not load graph: {e}")
