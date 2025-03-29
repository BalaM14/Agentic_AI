START
  │
  ├── Import Libraries
  │     ├── from langchain_community.document_loaders import WebBaseLoader
  │     ├── from langchain.text_splitter import RecursiveCharacterTextSplitter
  │     ├── from langchain_openai import OpenAIEmbeddings
  │     ├── from langchain_community.vectorstores import FAISS
  │     ├── from langchain.chains import create_retrieval_chain
  │     └── from langchain_core.documents import Document
  │
  ├── Load Web Document
  │     └── loader = WebBaseLoader("URL")
  │         document = loader.load()
  │
  ├── Document Splitting
  │     └── text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
  │         documents = text_splitter.split_documents(document)
  │
  ├── Embedding Generation
  │     └── embeddings = OpenAIEmbeddings()
  │
  ├── Vector Store Creation
  │     └── vector_store = FAISS.from_documents(documents, embeddings)
  │
  ├── Create Retriever
  │     └── retriever = vector_store.as_retriever()
  │
  ├── Create Document Chain
  │     └── document_chain.invoke({
  │           "input": "Note that ChatModels receive message",
  │           "context": [
  │               Document(page_content="Your detailed content here...")
  │           ]
  │         })
  │
  ├── Retrieval Chain Creation
  │     └── retrieval_chain = create_retrieval_chain(retriever, document_chain)
  │
  ├── Query Search
  │     ├── query = "Sample query text"
  │     ├── result = vector_store.similarity_search(query)
  │     └── answer = retrieval_chain.run(query)
  │
  └── Display Results
        ├── print(result[0].page_content)
        └── print(answer)
  │
END
