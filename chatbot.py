from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
import gradio as gr

# import the .env file
from dotenv import load_dotenv
load_dotenv()

# configuration
DATA_PATH = r"Data"
CHROMA_PATH = r"chroma_db"

embeddings_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# initiate the model
llm = ChatGoogleGenerativeAI(temperature=0.5, model="gemini-2.5-flash")

# connect to the chromadb
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)

# Set up the vectorstore to be the retriever
num_results = 5
retriever = vector_store.as_retriever(search_kwargs={'k': num_results})

# call this function for every message added to the chatbot
def stream_response(message, history):
    # retrieve the relevant chunks based on the question asked
    docs = retriever.invoke(message)

    # add all the chunks to 'knowledge'
    knowledge = ""
    for doc in docs:
        knowledge += doc.page_content + "\n\n"

    # make the call to the LLM (including prompt)
    if message is not None:
        partial_message = ""
        rag_prompt = f"""
        You are an assistent which answers questions based on knowledge which is provided to you.
        While answering, you don't use your internal knowledge,
        but solely the information in the "The knowledge" section.
        You don't mention anything to the user about the povided knowledge.

        The question: {message}

        Conversation history: {history}

        The knowledge: {knowledge}
        """

        # use invoke instead of stream to avoid Gemini content type issues
        response = llm.invoke(rag_prompt)
        content = response.content
        if isinstance(content, list):
            content = "".join(
                c if isinstance(c, str) else c.get("text", "")
                for c in content
            )
        yield content

# initiate the Gradio app
chatbot = gr.ChatInterface(stream_response, title="Gemini RAG Chatbot", textbox=gr.Textbox(placeholder="Send to the LLM...",
    container=False,
    autoscroll=True,
    scale=7),
)

# launch the Gradio app
chatbot.launch()
