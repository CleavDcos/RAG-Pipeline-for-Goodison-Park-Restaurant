from cProfile import label

import gradio as gr
from openai import OpenAI  

# Let's import os, which stands for "Operating System"
import os

# This will be used to load the API key from the .env file
from dotenv import load_dotenv
load_dotenv()

# Get the OpenAI API keys from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Let's configure the OpenAI Client using our key
openai_client = OpenAI(api_key=openai_api_key)
print("OpenAI client successfully configured.")



from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQAWithSourcesChain

# Define the path to your data file
# Ensure 'eleven_madison_park_data.txt' is in the same folder as this notebook
DATA_FILE_PATH = "eleven_madison_park_data.txt"


# -----Let's load Eleven Madison Park Restaurant data, which has been scraped from their website-----
# The data is saved in "eleven_madison_park_data.txt", Langchain's TextLoader makes this easy to read
print(f"Attempting to load data from: {DATA_FILE_PATH}")

# Initialize the TextLoader with the file path and specify UTF-8 encoding
# Encoding helps handle various characters correctly
loader = TextLoader(DATA_FILE_PATH, encoding = "utf-8")

# Load the document(s) using TextLoader from LangChain, which loads the entire file as one Document object
raw_documents = loader.load()
print(f"Successfully loaded {len(raw_documents)} document(s).")

#----Splitting----
#Now taking that large document and splitting it into smaller chunks that are easier to work with
# We will use RecursiveCharacterTextSplitter from LangChain, which splits text based on character count

print("\nSplitting the loaded document into smaller chunks...")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#chunk size is size of each chunk and overlap is how much the chunks overlap to maintain context,

documents = text_splitter.split_documents(raw_documents)
if not documents:
 raise ValueError("No documents were created after splitting. Please check the input data and splitting parameters.")
print(f"Document splitting complete. Created {len(documents)} chunks.")


#---Embeddings and Vector Store---
#Now we will create embeddings for each chunk of text and store them in a vector database for efficient retrieval
print("\nCreating embeddings and storing them in a vector database...")

#Create an instance of OpenAIEmbeddings, which will generate vector representations of our text chunks
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
print("OpenAI embeddings are initialized")


# Now the chunks from 'documents' are being converted to a vector using the 'embeddings' model
# The vectors are then stored as a vector in ChromaDB
# You could add `persist_directory="./my_chroma_db"` to save it to disk
# You will need to specify: (1) The list of chunked Document objects and (2) The embedding model to use
#Create Vector DB store(Chroma) to store the embeddings, which allows for efficient similarity search
vector_store = Chroma.from_documents(documents = documents, embedding=embeddings, persist_directory="./chroma_db")
print("Embeddings created and stored in the vector database successfully.")

vector_count = vector_store._collection.count()
if vector_count ==0:
 raise ValueError("No vectors were stored in the vector database. Please check the embedding and storage process.")

#---Testing the Retreieval---
# Let's perform a similarity search in our vector store
print("\n--- Testing Similarity Search in Vector Store ---")
test_query = "What different menus are offered?"
print(f"Searching for documents similar to: '{test_query}'")

# Perform a similarity search in the vector store using the test query, k=2 implies top 2 similar searches
try:
 similar_docs = vector_store.similarity_search(test_query, k=2)
 print(f"Found {len(similar_docs)} similar document(s).")

 #Display Snippets of the retrieved documents and their sources
 for i, doc in enumerate(similar_docs):
        print(f"\n--- Document {i+1} ---")
        # Displaying the first 700 chars for brevity
        content_snippet = doc.page_content[:700].strip() + "..."
        source = doc.metadata.get("source", "Unknown Source")  # Get source from metadata
        print(f"Content Snippet: {content_snippet}")
        print(f"Source: {source}")
except Exception as e:
 print(f"An error occurred during similarity search: {e}")

#---Building and Testing the RAG Chain using LangChain---

#We will be using RetrievalQAWithSourcesChain from LangChain, which allows us to build a question-answering system that retrieves relevant documents and provides sources for the answers.
#this chain combines a retriever(fetch vector store) and LLM(generate answer based on question, we'll use OpenAI)

#--1.Define the Retriever--
#retriever is used to fetch docs from vector store
#we configure it toi find top "k" documents
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
print("\nRetriever configured to fetch top 3 relevant documents.")

#--2Define the LLM--(using OpenAI's Chat model)
llm = ChatOpenAI(
    temperature=0.7,
    openai_api_key=openai_api_key
)
print("OpenAI LLM initialized with temperature 0.7.")

# --- 3. Create the RetrievalQAWithSourcesChain ---
# This chain type is designed specifically for Q&A with source tracking.
# chain_type="stuff": Puts all retrieved text directly into the prompt context.
#                      Suitable if the total text fits within the LLM's context limit.
#                      Other types like "map_reduce" handle larger amounts of text.
qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm=llm,
                                                       chain_type="stuff",
                                                       retriever=retriever,
                                                       return_source_documents=True, #request actual document objects used
                                                       verbose=True)#set to true to see langchains internal steps

print("\nRetrievalQAWithSourcesChain created successfully.")


# --- Test the Full Chain ---
print("\n--- Testing the Full RAG Chain ---")
chain_test_query = "What kind of food does Eleven Madison Park serve?"
print(f"Query: {chain_test_query}")

# Run the query through the chain. Use invoke() for Langchain >= 0.1.0
# The input must be a dictionary, often with the key 'question'.
try:
    result = qa_chain.invoke({"question": chain_test_query})

    # Print the answer and sources from the result dictionary
    print("\n--- Answer ---")
    print(result.get("answer", "No answer generated."))

    print("\n--- Sources ---")
    print(result.get("sources", "No sources identified."))

    # Optionally print snippets from the source documents returned
    if "source_documents" in result:
        print("\n--- Source Document Snippets ---")
        for i, doc in enumerate(result["source_documents"]):
            content_snippet = doc.page_content[:250].strip()
            print(f"Doc {i+1}: {content_snippet}")

except Exception as e:
    print(f"\nAn error occurred while running the chain: {e}")
    # Consider adding more specific error handling if needed



#----Implement Gradio for UI----

#Gradio helper
def answer_question(user_query):
   """
   Function to answer user questions using the RAG chain.
   """
   print("checking if query is proper...")
   if not user_query or user_query.strip() == "":
      return "please enter a valid question"
   
   try:
      result = qa_chain.invoke({"question": user_query})
      answer= result.get("answer", "No answer")
      sources= result.get("sources", "NO sources")
   
      print(f"--->Answer generated: {answer} ")
      print(f"--->Sources identified: {sources} ")

      return answer.strip(), sources

      
   except Exception as e:
            error_message = f"An error occurred: {e}"
            print(f"--> Error during chain execution: {error_message}")
        # Return error message to the user interface
            return error_message, "Error occurred"


print("\n setting up gradio interface...")

with gr.Blocks(theme=gr.themes.Soft()) as demo:
   # Title and description for the app
    gr.Markdown(
        """
        # Eleven Madison Park - AI Q&A Assistant 💬
        Ask questions about the restaurant based on its website data.
        The AI provides answers and cites the source document.
        *(Examples: What are the menu prices? Who is the chef? Is it plant-based?)*
        """
    )

    question_input = gr.Textbox(
        label="enter your question: ",
        placeholder="e.g. What kind of food does Eleven Madison Park serve?",
        lines=2,
    )

    #Outputs
    with gr.Row():

        answer_output = gr.Textbox(label="Answer:",interactive=False,
                                   lines=5)
        source_output = gr.Textbox(label="Sources:",interactive=False,
                                   lines=2)
    #Buttons
    submit_btn=gr.Button("Get Answer", variant="primary")
    clear_button=gr.Button("Clear All")  


      # Add some example questions for users to try
    gr.Examples(
        examples=[
            "What are the different menu options and prices?",
            "Who is the head chef?",
            "What is Magic Farms?"],
        inputs=question_input,  # Clicking example loads it into this input
        # We could potentially add outputs=[answer_output, sources_output] and cache examples
        # but that requires running the chain for each example beforehand.
        cache_examples=False,  # Don't pre-compute results for examples for simplicity
    )

    submit_btn.click(
        fn=answer_question,
        inputs=question_input,
        outputs=[answer_output, source_output]
    )

    clear_button.click(
    lambda: ("", "", ""),
    inputs=[],
    outputs=[question_input, answer_output, source_output]
    )
print("\nLaunching Gradio app... (Stop the kernel or press Ctrl+C in terminal to quit)")
# demo.launch() # Launch locally in the notebook or browser
demo.launch()
   

