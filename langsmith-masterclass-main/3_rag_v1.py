# pip install -U langchain langchain-community langchain-groq \
# faiss-cpu pypdf sentence-transformers python-dotenv

import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Free embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# Free LLM
from langchain_groq import ChatGroq

from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)

from langchain_core.output_parsers import StrOutputParser

# LangSmith project
os.environ['LANGCHAIN_PROJECT'] = 'langsmith-demo-RAG-v1'

load_dotenv()

PDF_PATH = "F:\\langgraph-tutorials-main\\langsmith-masterclass-main\\islr.pdf"

# 1) Load PDF
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()

# 2) Split documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

splits = splitter.split_documents(docs)

# 3) Free Embeddings Model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4) Create FAISS vector DB
vs = FAISS.from_documents(splits, embeddings)

retriever = vs.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# 5) Prompt
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Answer ONLY from the provided context. "
        "If answer is not found in context, say 'I don't know'."
    ),
    (
        "human",
        "Question: {question}\n\nContext:\n{context}"
    )
])

# 6) Free Open-Source LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7
)

# Helper function
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 7) Parallel chain
parallel = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

# 8) Final chain
chain = parallel | prompt | llm | StrOutputParser()

# 9) Chat loop
print("PDF RAG ready. Ask a question (Ctrl+C to exit).")

while True:
    q = input("\nQ: ")

    if q.lower() in ["exit", "quit"]:
        break

    ans = chain.invoke(q.strip())

    print("\nA:", ans)