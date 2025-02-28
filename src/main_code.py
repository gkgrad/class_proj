import os
import shutil
import sqlite3
import numpy as np
import torch
from docling.document_converter import DocumentConverter
from sentence_transformers import SentenceTransformer
import PyPDF2
from pdf2image import convert_from_path
from PIL import Image
import base64
import io
import json
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from docx import Document
from proj import config

# Load models
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
MODEL = config["models"]["english-sentence-encoder"]
embedder = SentenceTransformer(MODEL)
results_dir  = config["paths"]["results"]

class DocumentDatabase:
    def __init__(self, db_name):
        self.db_name = db_name
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.create_tables()

    def connect(self):
        try:
            self.conn = sqlite3.connect(self.db_name)
            return self.conn
        except sqlite3.Error as e:
            print(f"Connection error: {e}")
            return False

    def create_tables(self):
        # Create DOCUMENT table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS DOCUMENT (
                DocumentID INTEGER PRIMARY KEY,
                FileName TEXT NOT NULL,
                DocumentType TEXT NOT NULL,
                PlainTextContent TEXT
            )
        ''')

        # Create IMAGE table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS IMAGE (
                ImageID INTEGER PRIMARY KEY,
                DocumentID INTEGER NOT NULL,
                ImageData BLOB,
                EmbeddingVector BLOB,
                FOREIGN KEY (DocumentID) REFERENCES DOCUMENT(DocumentID)
            )
        ''')

        # Create CHUNK/Embedding table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS CHUNK (
                ChunkID INTEGER PRIMARY KEY,
                DocumentID INTEGER NOT NULL,
                ChunkText TEXT NOT NULL,
                Embedding BLOB,  -- For storing binary embeddings
                EmbeddingJSON TEXT,  -- For storing JSON-formatted embeddings
                FOREIGN KEY (DocumentID) REFERENCES DOCUMENT(DocumentID)
            )
        ''')
        # Create METADATA table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS METADATA (
                MetadataID INTEGER PRIMARY KEY,
                DocumentID INTEGER NOT NULL,
                Key TEXT NOT NULL,
                Value TEXT NOT NULL,
                FOREIGN KEY (DocumentID) REFERENCES DOCUMENT(DocumentID)
            )
        ''')

        self.conn.commit()

    def is_file_processed(self, filename):
        self.cursor.execute('''
            SELECT * FROM DOCUMENT WHERE FileName = ?
        ''', (filename,))
        return self.cursor.fetchone() is not None
    
    def insert_documents(self, documents):
        document_ids = []
        for filename, document_type, plain_text_content in documents:
            if not self.is_file_processed(filename):
                self.cursor.execute('''
                    INSERT INTO DOCUMENT (FileName, DocumentType, PlainTextContent)
                    VALUES (?, ?, ?)
                ''', (filename, document_type, plain_text_content))
                self.conn.commit()
                document_ids.append(self.cursor.lastrowid)
            else:
                print(f"Skipping file: {filename} (already processed)")
        return document_ids

    def insert_chunks(self, document_id, chunks, embeddings):
        chunk_ids = []
        for chunk_text, embedding in zip(chunks, embeddings):
            # Convert numpy array to binary or JSON
            embedding_binary = embedding.tobytes()  # For BLOB
            embedding_json = json.dumps(embedding.tolist())  # For JSON
            
            self.cursor.execute('''
                INSERT INTO CHUNK (DocumentID, ChunkText, Embedding, EmbeddingJSON)
                VALUES (?, ?, ?, ?)
            ''', (document_id, chunk_text, embedding_binary, embedding_json))
            self.conn.commit()
            chunk_ids.append(self.cursor.lastrowid)
        return chunk_ids

    def insert_image(self, document_id, image_data, embedding_vector):
        self.cursor.execute('''
            INSERT INTO IMAGE (DocumentID, ImageData, EmbeddingVector)
            VALUES (?, ?, ?)
        ''', (document_id, image_data, embedding_vector))
        self.conn.commit()
        return self.cursor.lastrowid
        
    def retrieve_chunks(self, document_id):
        self.cursor.execute('''
            SELECT ChunkText FROM CHUNK WHERE DocumentID = ?
        ''', (document_id,))
        chunks = [row[0] for row in self.cursor.fetchall()]
        return chunks

    def insert_metadata(self, document_id, metadata):
        metadata_ids = []
        for key, value in metadata:
            self.cursor.execute('''
                INSERT INTO METADATA (DocumentID, Key, Value)
                VALUES (?, ?, ?)
            ''', (document_id, key, value))
            self.conn.commit()
            metadata_ids.append(self.cursor.lastrowid)
        return metadata_ids

    def get_document(self, document_id):
        self.cursor.execute('''
            SELECT * FROM DOCUMENT WHERE DocumentID = ?
        ''', (document_id,))
        return self.cursor.fetchone()

    def get_chunks(self, document_id):
        self.cursor.execute('''
            SELECT * FROM CHUNK WHERE DocumentID = ?
        ''', (document_id,))
        return self.cursor.fetchall()

    def get_embeddings(self):
        
        self.cursor.execute('''
            SELECT Embedding FROM CHUNK
        ''')
        embeddings = self.cursor.fetchall()
        # Convert binary embeddings back to numpy arrays
        embeddings = [np.frombuffer(embedding[0], dtype=np.float32) for embedding in embeddings]
        return np.stack(embeddings)

    def get_metadata(self, document_id):
        self.cursor.execute('''
            SELECT * FROM METADATA WHERE DocumentID = ?
        ''', (document_id,))
        return self.cursor.fetchall()

    def get_image(self, image_id):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Query the database for the embedding
        cursor.execute('''
            SELECT EmbeddingVector
            FROM IMAGE
            WHERE ImageID = ?
        ''', (image_id,))

        result = cursor.fetchone()
        conn.close()
        if result is None:
            raise ValueError(f"No image found with ID {image_id}")

        # Convert the BLOB back to a numpy array
        embedding_blob = result[0]
        embedding = np.frombuffer(embedding_blob, dtype=np.float32)

        return embedding


    def generate_text_embeddings(self, chunks, document_id):
    
        if not chunks:
            print("❌ No chunks found in the database.")
            return None
        embeddings = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
       
        chunk_ids = self.insert_chunks(document_id, chunks, embeddings)
        print(chunk_ids)
        #embeddings = [np.frombuffer(embedding[0], dtype=np.float32) for embedding in embeddings]
        return embeddings

    def generate_image_embedding(self, images, document_id):

        inputs = clip_processor(images, return_tensors="pt")
        with torch.no_grad():
            image_features = clip_model.get_image_features(**inputs)
        embedding_vector = image_features.squeeze().numpy()
        image_id = insert_image(document_id, images, embedding_vector.tobytes())
        return image_id

    def close_connection(self):
        self.conn.close()

def extract_from_pdf(file_path):

    # Use Docling to extract text
   # text = docling.convert(file_path, from_format="pdf", to_format="text")
    converter = DocumentConverter()
    result = converter.convert(file_path)
    text = result.document.export_to_markdown()

    images = convert_from_path(file_path)
    encoded_images = []
    for image in images:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        encoded_images.append(encoded_image)

    return text, encoded_images

def extract_from_docx(file_path):
    doc = docx.Document(file_path)
    text = []
    for para in doc.paragraphs:
        text.append(para.text)

    images = []
    for rel in doc._rels.values():
        if rel.reltype == docx.image.image.Image:
            image_path = rel.target_ref
            with open(image_path, "rb") as file:
                image_data = file.read()
                encoded_image = base64.b64encode(image_data).decode("utf-8")
                images.append(encoded_image)

    return " ".join(text), images

def get_documents():
    if not os.path.exists("folder1"):
        print(f"❌ Folder {"folder1"} does not exist.")
        return

    document_files = [f for f in os.listdir("folder1") if f.lower().endswith(('.pdf', '.doc', '.docx', '.txt'))]

    if not document_files:
        print(f"📂 No documents found in {"folder1"}.")
        return
    return document_files

def chunk_text(text, chunk_size=500):
    splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
    )

    # Split the text into chunks
    chunks = splitter.split_text(text)
    return chunks

def create_faiss_hnsw_index(embeddings:np.ndarray, hnsw_m: int=32):
    """    Creates an HNSW index with M neighbors per node.
    Higher M leads to better recall but increases memory and indexing time.

    Args:
        embeddings (numpy.ndarray): embeddings to be indexed
        hnsw_m (int, optional): the M parameter. Defaults to 32.
    Returns:
        _type_: Faiss HNSW index
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dimension, hnsw_m)  # HNSW with L2 distance
    index.hnsw.efConstruction = 200  # Control search quality (higher = better recall)
    index.add(embeddings)  # Add vectors to the index
    return index

def load_faiss_index(filename):
    return faiss.read_index(filename)

def process_documents():

    db = DocumentDatabase('document_database.db')
    faiss_index_file = f"{results_dir}/faiss_index_hnsw.bin"

    document_files = get_documents()
    for doc_file in document_files:
        file_path = os.path.join("folder1", doc_file)
        file_extension = os.path.splitext(doc_file)[1].lower()

        print(f"📄 insert document in db: {doc_file}")
        document = [(file_path, file_extension, "")]

        # Insert a document
        #document_id = db.insert_documents(document)
        #print(f"Inserted document with ID: {document_id}")

        
    # Parsing

        if file_extension == '.pdf':
            text, images = extract_from_pdf(file_path)
        elif file_extension in ['.doc', '.docx']:
            text, images = extract_from_docx(file_path)
        elif file_extension == '.txt':
            text, images = extract_text_from_txt(file_path)
        else:
            print(f"⚠️ Unsupported file type: {file_extension}")
            continue
        #print(text)
         # Insert a document
        document_id = db.insert_documents(document)
        print(f"Inserted document with ID: {document_id}")
        if document_id == []:
            return


    # Chunking and indexing
        chunks = chunk_text(text)
        print(chunks[0], chunks[1], chunks[150])
        import pdb
        pdb.set_trace()
        text_embeddings = db.generate_text_embeddings(chunks, document_id[0])
        print(text_embeddings)
        #image_embeddings = db.generate_image_embeddings(images, document_id[0])

        print(f"✅ Stored {len(chunks)} chunks and embeddings for {file_path} in SQLite!")
    
    # creating fiass index
        if len(text_embeddings) != 0:
            faiss_index = create_faiss_hnsw_index(text_embeddings)
            print(faiss_index)
            print(faiss_index.ntotal)
            faiss.write_index(faiss_index, faiss_index_file)
            print(f"✅ Stored {faiss_index} for {document_id[0]} in {faiss_index_file}!")
    

if __name__ == "__main__":
   
    process_documents()



    