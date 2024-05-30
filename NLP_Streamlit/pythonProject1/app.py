import streamlit as st
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle

class SentenceTransformersEmbeddings:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=True)

st.set_page_config(page_title="NLP Dashboard", layout="wide")

url = st.sidebar.text_input("Enter your URL")
button_clicked = st.sidebar.button('Summarize')
file_path = "faiss_store_sentence_transformers.pkl"
main_placeholder = st.empty()
other_texts= ""

if button_clicked:
    loader = UnstructuredURLLoader(urls=[url])
    main_placeholder.text("Loading data.........")
    data = loader.load()

    main_placeholder.text("Splitting Data.........")

    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ".", "!"],
                                                   chunk_size=500)

    text_splitter = text_splitter.split_documents(data)

    model = SentenceTransformer('all-MiniLM-L6-v2')

    texts = [text_splitter[i].page_content for i in range(len(text_splitter))]
    other_texts = texts

    main_placeholder.text("Encoding data......")

    embeddings = model.encode(texts)

    embeddings = np.array(embeddings)

    # Create FAISS index
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)  # L2 distance (Euclidean distance)
    faiss_index.add(embeddings)


    with open(file_path, "wb") as f:
        pickle.dump(faiss_index, f)

query = main_placeholder.text_input("Enter your query")

if query:
    embeddings_model = SentenceTransformersEmbeddings()
    my_result = ""
    with open(file_path, "rb") as f:
        faiss_index = pickle.load(f)


        def search_faiss_index(query, index, texts, embeddings_model, k=5):
            query_embedding = embeddings_model.embed_documents([query])[0]
            query_embedding = np.expand_dims(query_embedding, axis=0)
            distances, indices = index.search(query_embedding, k)
            return [(texts[i], distances[0][j]) for j, i in enumerate(indices[0])]


        results = search_faiss_index(query, faiss_index, other_texts, embeddings_model)

        i = 0

        for result, distance in results:
            if i == 0:
                st.write("Answers...")
                st.write(result)
                i+=1
            else:
                break

