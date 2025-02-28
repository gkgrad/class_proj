import os
import numpy as np
from proj import config
from tqdm import tqdm
from main_code import DocumentDatabase
from docling.document_converter import DocumentConverter
from sentence_transformers import SentenceTransformer
from main_code import load_faiss_index
import pprint

class QueryDocument:
    def __init__(self):
        MODEL = config["models"]["english-sentence-encoder"]
        embedder = SentenceTransformer(MODEL)
        results_dir = config["paths"]["results"]
        for file in os.listdir(results_dir):
            faiss_index_file = f"{results_dir}/{file}"
        self.embedder = embedder
        self.faiss_index_file = faiss_index_file
        self.results_dir = results_dir
        self.db = DocumentDatabase('document_database.db')
    
    
    def search_hnsw_index(self, query, index, k=5):
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)
        #query_embedding = np.ascontiguousarray(query_embedding, dtype='float32')
        #print("*********************************")
        #print(query_embedding.shape)
        index.hnsw.efSearch = 128  # Adjust tradeoff between speed and recall
        distances, indices = index.search(query_embedding, k)
        
        # Assuming the indices correspond to rows in your IMAGE or CHUNK table
        conn = self.db.connect()
        cursor = conn.cursor()

        results = []
        for idx in range(len(indices[0])):
            index_id = indices[0][idx]  # Get the index within the Faiss index
            #print("INDEX_ID", index_id)
            # Assuming the indices correspond to the order in which embeddings were added
            cursor.execute('''
                SELECT ChunkID, DocumentID, ChunkText FROM CHUNK
            ''')
            all_chunks = cursor.fetchall()
            if index_id < len(all_chunks):
                chunk_id, document_id, chunk_text = all_chunks[index_id]

                cursor.execute('''
                    SELECT FileName FROM DOCUMENT WHERE DocumentID = ?
                ''', (document_id,))
                document_name = cursor.fetchone()[0]

                results.append({
                    'type': 'text',
                    'document_name': document_name,
                    'chunk_text': chunk_text,
                    'distance': distances[0][idx]
                })
            else:
                print(f"Warning: Index {index_id} out of bounds for chunks.")

        conn.close()
        return results


    # Example usage:
    def search_and_print_results(self, query):
        print(query)
        #
        # query = "what is Similarity measures"
        index = load_faiss_index(self.faiss_index_file)  # Load your HNSW index
        #print("#########################")
        #print(index.d)  # This should m
        #print(index.ntotal) 
        results = self.search_hnsw_index(query, index)
        #print(results)

        if results:

            print("Search Results:")

            for i, result in enumerate(results):
                
                print(f"\nResult {i+1}:")
               
                print(f"  Document: {result['document_name']}")

                print(f"  Chunk Text: {result['chunk_text']}")

                print(f"  Score: {result['distance']:.4f}") # Format the distance

                print("-" * 20)  # Separator

        else:

            print("No results found.")
    
def main():
    query = QueryDocument()
    search_query = "risk."
    print(query.search_and_print_results(search_query))
    

if __name__ == "__main__":
    main()