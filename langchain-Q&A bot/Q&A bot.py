import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama


def load_markdown_documents(folder_path):
    docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".md"):
            loader = TextLoader(os.path.join(folder_path, filename), encoding='utf-8')
            docs.extend(loader.load())
    print(f"📄 加载完成：{len(docs)} 个 Markdown 文档")
    return docs


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)

def build_vector_store(docs):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings)


def build_qa_chain(vector_store):
    llm = Ollama(model="deepseek-r1:1.5b")
    return RetrievalQA.from_chain_type(llm=llm, retriever=vector_store.as_retriever())

def main():
   
    docs = load_markdown_documents("docs")
    chunks = split_documents(docs)

    
    vector_store = build_vector_store(chunks)

    
    qa = build_qa_chain(vector_store)

    
    while True:
        query = input("\n❓请输入问题（exit 退出）：")
        if query.lower() in ['exit', 'quit']:
            break
        result = qa.run(query)
        print(f"✅ 答案：{result}")

if __name__ == "__main__":
    main()
