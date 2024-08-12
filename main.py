from fastapi import FastAPI, Request
import nltk
from langchain_community.llms import Ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader, UnstructuredURLLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.agents import Tool
import uvicorn

app = FastAPI()
nltk.download('averaged_perceptron_tagger')

chat_history = []
folder_path = "knowledge"
cached_llm = Ollama(model="gemma:2b")

urls = [
    "https://newwavesolution.com/",
    "https://newwavesolution.com/about-us/",
    "https://newwavesolution.com/services/",
    "https://newwavesolution.com/industries/",
    "https://newwavesolution.com/technologies/",
    "https://newwavesolution.com/solutions/",
    "https://newwavesolution.com/developers/",
    "https://newwavesolution.com/portfolio/",
    "https://newwavesolution.com/blog/",
]

loader = UnstructuredURLLoader(urls=urls)
model_name = "all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200, separators=["\n\n", "\n", " ", ""])

raw_prompt = PromptTemplate.from_template(""" 
    You are an assistant and your name is BOT AI that helps visitors find the best answer.
    If you can't find the answer from the context, ask more questions. Do not make up your own questions.
    Use the questioner's language to answer accurately.
    {input}
    Context: {context}
    Answer: 
""")


@app.post("/ai")
async def aiPost(request: Request):
    json_content = await request.json()
    query = json_content.get("query")
    response = cached_llm.invoke(query)
    return {"answer": response}


@app.post("/ask")
async def askPost(request: Request):
    json_content = await request.json()
    query = json_content.get("query")
    vector_store = Chroma(persist_directory=folder_path, embedding_function=embeddings)
    retriever = vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 5, "score_threshold": 0.1})
    retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("human", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation."),
    ])
    history_aware_retriever = create_history_aware_retriever(llm=cached_llm, retriever=retriever, prompt=retriever_prompt)
    document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
    retrieval_chain = create_retrieval_chain(history_aware_retriever, document_chain)
    result = retrieval_chain.invoke({"input": query})
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=result["answer"]))
    sources = [{"source": doc.metadata["source"], "page_content": doc.page_content} for doc in result["context"]]
    return {"answer": result["answer"], "sources": sources}


@app.post("/web")
async def webPost():
    data = loader.load()
    texts = text_splitter.split_documents(data)
    db = Chroma.from_documents(texts, embeddings, persist_directory=folder_path)
    db.persist()
    return {"status": "Successfully Uploaded"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
