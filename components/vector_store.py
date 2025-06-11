import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer
import os


class VectorStore:
    def __init__(self):
        # Ensure database directory exists
        os.makedirs("./database/chroma_db", exist_ok=True)

        self.client = chromadb.PersistentClient(path="./database/chroma_db")
        self.collection_name = "document_collection"
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Initialize collection
        try:
            self.collection = self.client.get_collection(self.collection_name)
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )

    def add_document(self, doc_data):
        """Add document chunks to vector store"""
        try:
            documents = []
            metadatas = []
            ids = []

            for chunk in doc_data['chunks']:
                chunk_text = chunk['content']

                documents.append(chunk_text)
                metadatas.append({
                    'document_id': doc_data['id'],
                    'document_name': doc_data['name'],
                    'document_type': doc_data['type'],
                    'chunk_id': chunk['chunk_id'],
                    'upload_date': doc_data['upload_date'].isoformat(),
                    'word_count': doc_data['metadata']['word_count']
                })
                ids.append(f"{doc_data['id']}_{chunk['chunk_id']}")

            # Add to collection
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

            return True

        except Exception as e:
            st.error(f"Error adding document to vector store: {str(e)}")
            return False

    def search(self, query, max_results=10, threshold=0.7):
        """Search for relevant document chunks"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=max_results,
                include=['documents', 'metadatas', 'distances']
            )

            search_results = []

            if results['documents'] and results['documents'][0]:
                for doc, metadata, distance in zip(
                        results['documents'][0],
                        results['metadatas'][0],
                        results['distances'][0]
                ):
                    similarity = 1 - distance

                    if similarity >= threshold:
                        search_results.append({
                            'content': doc,
                            'document_id': metadata['document_id'],
                            'document_name': metadata['document_name'],
                            'similarity': similarity,
                            'metadata': metadata
                        })

            return sorted(search_results, key=lambda x: x['similarity'], reverse=True)

        except Exception as e:
            st.error(f"Error searching vector store: {str(e)}")
            return []
