import json
import os
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import (
    TextLoader,
    CSVLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
)
from dotenv import load_dotenv
import os

load_dotenv()


llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0,
)

# Changing the embedding model requires deleting ./persist and re-indexing:
# vectors from different models are not comparable.
embeddings = GoogleGenerativeAIEmbeddings(
    model=os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001"),
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

# Tries to cut by paragraph, then line, then sentence, so each rule or section stays whole.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)


def load_document(file_path: str) -> list[Document]:
    """
    Load a document from a file path.
    Supports .txt, .pdf, .docx, .csv, .html, .md and .json files.

    Args:
    file_path (str): Path to the document file.

    Returns:
    list[Document]: A list of Document objects.

    Raises:
    ValueError: If the file type is not supported.
    """
    _, file_extension = os.path.splitext(file_path)

    if file_extension == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    elif file_extension == ".pdf":
        loader = PyPDFLoader(file_path)
    elif file_extension == ".docx":
        loader = Docx2txtLoader(file_path)
    elif file_extension == ".csv":
        loader = CSVLoader(file_path, encoding="utf-8")
    elif file_extension == ".html":
        loader = UnstructuredHTMLLoader(file_path)
    elif file_extension == ".md":
        loader = UnstructuredMarkdownLoader(file_path)
    elif file_extension == ".json":
        return load_faq_json(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

    return loader.load()


def load_faq_json(file_path: str) -> list[Document]:
    """
    Load a FAQ file: a JSON list of objects with "pregunta" and "respuesta".
    Each question-answer pair becomes its own Document, so the question always
    stays next to its answer.
    """
    with open(file_path, encoding="utf-8") as f:
        faqs = json.load(f)

    return [
        Document(
            page_content=f"Pregunta: {faq['pregunta']}\nRespuesta: {faq['respuesta']}",
            metadata={"source": file_path, "id": faq.get("id"), "tema": faq.get("tema")},
        )
        for faq in faqs
    ]


def create_collection(collection_name, documents):
    """
    Create a new Chroma collection from the given documents.

    Args:
    collection_name (str): The name of the collection to create.
    documents (list): A list of documents to add to the collection.

    Returns:
    None

    This function splits the documents into texts, creates a new Chroma collection,
    and persists it to disk.
    """
    # Split the documents into smaller text chunks
    texts = text_splitter.split_documents(documents)
    persist_directory = "./persist"

    # Create a new Chroma collection from the text chunks
    try:
        vectordb = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name,
        )
    except Exception as e:
        print(f"Error creating collection: {e}")
        return None

    return vectordb


def load_collection(collection_name):
    """
    Load an existing Chroma collection.

    Args:
    collection_name (str): The name of the collection to load.

    Returns:
    Chroma: The loaded Chroma collection.

    This function loads a previously created Chroma collection from disk.
    """
    persist_directory = "./persist"
    # Load the Chroma collection from the specified directory
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name=collection_name,
    )

    return vectordb


def load_retriever(collection_name, score_threshold: float = 0.6):
    """
    Create a retriever from a Chroma collection with a similarity score threshold.

    Args:
    collection_name (str): The name of the collection to use.
    score_threshold (float): The minimum similarity score threshold for retrieving documents.
                           Documents with scores below this threshold will be filtered out.
                           Defaults to 0.6.

    Returns:
    Retriever: A retriever object that can be used to query the collection with similarity
              score filtering.

    This function loads a Chroma collection and creates a retriever from it that will only
    return documents meeting the specified similarity score threshold.
    """
    # Load the Chroma collection
    vectordb = load_collection(collection_name)
    # Create a retriever from the collection with specified search parameters
    retriever = vectordb.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": score_threshold},
    )
    return retriever


ROLE_PROMPT = """
Actúa como "Eco", un agente de servicio al cliente amable y preciso de EcoMarket,
una tienda en línea de productos sostenibles. Tuteas al cliente.
Respondes ÚNICAMENTE con la información de EcoMarket que aparece entre >>>>> y <<<<<.
Nunca inventas productos, precios, plazos, políticas ni estados de pedidos.
"""

INSTRUCTION_PROMPT = """
Información de EcoMarket:
>>>>>
{context}
<<<<<

Consulta del cliente: {question}

Sigue estos pasos:
1. Busca en la información de EcoMarket lo que responde la consulta.
2. Si la información no alcanza para responder, dilo con amabilidad, explica que no
   tienes esa información y ofrece comunicar al cliente con un agente humano.
   No completes la respuesta con conocimiento general.
3. Si el cliente pregunta por el estado de un pedido específico, explica que desde este
   chat no puedes consultar pedidos y ofrece comunicarlo con un agente humano.
4. Si la información sí alcanza, responde de forma clara y directa. Si es una regla con
   excepciones (por ejemplo, productos dañados), menciona la excepción.

Responde en español, en máximo 120 palabras, y cierra preguntando si puedes ayudar en
algo más.
"""

# Few-shot: a normal case and a case where the context does not have the answer
POSITIVE_EXAMPLE = """
Información de EcoMarket:
>>>>>
El kit de jabones artesanales tiene garantía de 3 meses por defectos de fabricación.
<<<<<

Consulta del cliente: ¿El kit de jabones tiene garantía?
"""

POSITIVE_OUTPUT = """
¡Hola! 🌱 Sí, el kit de jabones artesanales tiene una garantía de **3 meses** por
defectos de fabricación. Si notas algún defecto dentro de ese tiempo, escríbenos y lo
revisamos.

¿Te puedo ayudar con algo más?
"""

NEGATIVE_EXAMPLE = """
Información de EcoMarket:
>>>>>
El envío es gratis en compras desde $150.000.
<<<<<

Consulta del cliente: ¿Venden paneles solares para techo?
"""

NEGATIVE_OUTPUT = """
¡Hola! Gracias por escribirnos. 🌱 No tengo información sobre paneles solares para
techo en nuestro catálogo, así que no quiero darte una respuesta equivocada.

Si quieres, puedo comunicarte con uno de nuestros agentes para que lo revise contigo.
¿Te puedo ayudar con algo más?
"""


def format_docs(docs: list[Document]) -> str:
    """Join the retrieved chunks into a single text block for the prompt."""
    if not docs:
        return "(No se encontró información relacionada.)"
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def generate_answer_from_context(retriever, question: str):
    """
    Ask a question and get an answer based on the provided context.

    Args:
        retriever: A retriever object to fetch relevant context.
        question (str): The question to be answered.

    Returns:
        str: The answer to the question based on the retrieved context.
    """
    # Same structure as Taller 1: role, few-shot examples, delimiters and steps
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ROLE_PROMPT.strip()),
            ("human", POSITIVE_EXAMPLE.strip()),
            ("ai", POSITIVE_OUTPUT.strip()),
            ("human", NEGATIVE_EXAMPLE.strip()),
            ("ai", NEGATIVE_OUTPUT.strip()),
            ("human", INSTRUCTION_PROMPT.strip()),
        ]
    )

    # Create a RAG (Retrieval-Augmented Generation) chain
    # This chain retrieves context, passes through the question,
    # formats the prompt, and generates an answer using the language model
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    # Invoke the RAG chain with the question and return the generated content
    return rag_chain.invoke(question).content


def add_documents_to_collection(vectordb, documents):
    """
    Add documents to the vector database collection.

    Args:
        vectordb: The vector database object to add documents to.
        documents: A list of documents to be added to the collection.

    This function splits the documents into smaller chunks, adds them to the
    vector database, and persists the changes.
    """

    # Split the documents into smaller text chunks
    texts = text_splitter.split_documents(documents)

    # Add the text chunks to the vector database
    vectordb.add_documents(texts)

    return vectordb
