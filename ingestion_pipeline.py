import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
import chromadb
import hashlib

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
def get_docs_hash():
    docsPath = Path("docs")

    hasher = hashlib.sha256()
    for filePath in sorted(docsPath.glob("*.txt")):
        hasher.update(filePath.name.encode())
        hasher.update(filePath.read_bytes())

    return hasher.hexdigest()

def load_documents():
    docsPath = Path("docs")
    documents = []

    for filePath in docsPath.glob("*.txt"):
        content = filePath.read_text(encoding="utf-8")
        documents.append({"source": str(filePath), "content": content})

    return documents
def split_documents(documents, chunk_size = 1000):
    chunks = []
    for doc in documents:
        text = doc["content"]
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i+chunk_size]
            chunks.append({"source": doc["source"], "content": chunk})
    return chunks
def embedding_chunks(chunks):
    texts = []
    for chunk in chunks:
        texts.append(chunk["content"])
    response = client.embeddings.create(
    input=texts, model="text-embedding-3-small"
)
    embeddingsChunk = []
    for i, chunk in enumerate(chunks):
        embeddingsChunk.append({"source": chunk["source"], "content": chunk["content"], "embedding": response.data[i].embedding})
    return embeddingsChunk


    
def main():
    chroma_client = chromadb.PersistentClient(path="db/chroma")
    hashPath = Path("db/docs.hash")
    currentHash = get_docs_hash()
    oldHash = None
    if hashPath.exists():
        oldHash = hashPath.read_text()

    if currentHash != oldHash:
        documents = load_documents()
        chunks = split_documents(documents)
        embeddingChunks = embedding_chunks(chunks)
        try:
            chroma_client.delete_collection("documents")
        except:
            pass
        collection = chroma_client.create_collection(
            name="documents"
        )

        ids = []
        documents = []
        embeddings = []
        metadatas = []

        for i, chunk in enumerate(embeddingChunks):
            ids.append(f"chunk_{i}")
            documents.append(chunk["content"])
            embeddings.append(chunk["embedding"])
            metadatas.append({
                "source": chunk["source"]
            })

        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

        hashPath.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        hashPath.write_text(currentHash)

    else:
        collection = chroma_client.get_collection(
            "documents"
        )
    message = input("Digite a mensagem: ")
    response = client.embeddings.create(input=message,model="text-embedding-3-small")
    embeddingMessage = response.data[0].embedding

    results = collection.query(query_embeddings=[embeddingMessage],n_results=3 )
    documentsFound = results["documents"][0]
    context = "\n\n".join(documentsFound)

    aiChat = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Você é um assistente útil, responda às perguntas do usuário se baseando no contexto fornecido."
            },
            {
                "role": "user",
                "content": f"Contexto: {context}\nPergunta: {message}"
            }
        ],
        temperature=0.3
    )

    print(aiChat.choices[0].message.content)
if __name__ == "__main__":
    main()