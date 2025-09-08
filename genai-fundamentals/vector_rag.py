import os
from dotenv import load_dotenv
load_dotenv()

from neo4j import GraphDatabase
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.generation import GraphRAG
import logging, time

# Basic logging and environment validation
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# Silence Neo4j driver INFO notifications (e.g., JIT codegen messages)
logging.getLogger("neo4j").setLevel(logging.WARNING)
required_env_vars = ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "OPENAI_API_KEY"]
missing_env_vars = [name for name in required_env_vars if not os.getenv(name)]
if missing_env_vars:
    raise EnvironmentError(f"Missing env vars: {', '.join(missing_env_vars)}")

logging.info("Connecting to Neo4j at %s\n", os.getenv("NEO4J_URI"))

# Connect to Neo4j database
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"), 
    auth=(
        os.getenv("NEO4J_USERNAME"), 
        os.getenv("NEO4J_PASSWORD")
    )
)

# Verify connectivity and index exists
driver.verify_connectivity()
with driver.session() as session:
    index_check = session.run(
        "SHOW INDEXES YIELD name WHERE name = $name RETURN name",
        name="moviePlots",
    ).data()
    if not index_check:
        raise RuntimeError("Vector index 'moviePlots' not found. Run the reset.cypher step to create it.\n")

# Create embedder
embedder = OpenAIEmbeddings(model="text-embedding-ada-002")

# Create retriever
retriever = VectorRetriever(
    driver,
    index_name="moviePlots",
    embedder=embedder,
    return_properties=["title", "plot"],
)

# Create the LLM
llm = OpenAILLM(model_name="gpt-4o")

# Create GraphRAG pipeline
rag = GraphRAG(retriever=retriever, llm=llm)

# Search 
query_text = "Find me movies about war."

print(f"Query: {query_text}\n")

response = rag.search(
    query_text=query_text, 
    retriever_config={"top_k": 5}
)

print("\n", response.answer)

# Close the database connection
driver.close()