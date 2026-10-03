from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from uuid import uuid4
import time

# import the .env file
from dotenv import load_dotenv
load_dotenv()

# configuration
DATA_PATH = r"Data"
CHROMA_PATH = r"chroma_db"

# initiate the embeddings model
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# initiate the vector store
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)

# loading the PDF document
loader = PyPDFDirectoryLoader(DATA_PATH)
raw_documents = loader.load()

# splitting the document
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)

# creating the chunks
chunks = text_splitter.split_documents(raw_documents)

print(f"Total chunks to embed: {len(chunks)}")

# Add chunks in small batches to avoid hitting the free-tier rate limit
BATCH_SIZE = 50

for i in range(0, len(chunks), BATCH_SIZE):
    batch = chunks[i:i + BATCH_SIZE]
    batch_uuids = [str(uuid4()) for _ in range(len(batch))]
    vector_store.add_documents(documents=batch, ids=batch_uuids)
    print(f"Added batch {i // BATCH_SIZE + 1} ({len(batch)} chunks)")
    
    # Wait between batches to stay under rate limit
    if i + BATCH_SIZE < len(chunks):
        print("Waiting 60s to respect rate limit...")
        time.sleep(60)

print("Done! All chunks have been added to the vector store.")
