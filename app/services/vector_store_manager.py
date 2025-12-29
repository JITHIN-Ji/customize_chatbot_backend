import faiss
import numpy as np
import os
import json
from typing import List, Tuple, Optional, Dict, Any
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class VectorStoreManager:
    def __init__(self, user_id: str, index_path: str = settings.VECTOR_STORE_PATH, dimension: Optional[int] = None):
        
        index_path = os.path.join(index_path, user_id, "faiss_store")
        self.index_path_base = index_path 
        self.index_file_path = self.index_path_base + ".index"
        self.meta_file_path = self.index_path_base + ".meta.json"
        

        self.index_dir = os.path.dirname(self.index_path_base)
        os.makedirs(self.index_dir, exist_ok=True)
        
        self.index: Optional[faiss.Index] = None
        
        self.metadata_list: List[Dict[str, Any]] = [] 
                                                 

        if os.path.exists(self.index_file_path) and os.path.exists(self.meta_file_path):
            self.load_index()
        elif dimension is not None:
            logger.info(f"Initializing new FAISS index with dimension {dimension} at {self.index_file_path}")
            self.index = faiss.IndexFlatL2(dimension)
            self._save_metadata_list() 
        else:
            logger.warning(f"Vector store index {self.index_file_path} not found and dimension not provided for initialization.")
            

    def _initialize_index_if_needed(self, dimension: int):
        if self.index is None:
            logger.info(f"Initializing FAISS index with dimension {dimension}.")
            self.index = faiss.IndexFlatL2(dimension)
            self._save_metadata_list() 

    def add_embeddings(self, embeddings: np.ndarray, metadata_entries: List[Dict[str, Any]]):
        if embeddings.size == 0:
            logger.warning("No embeddings provided to add.")
            return
        
        if len(embeddings) != len(metadata_entries):
            logger.error(f"Mismatch between number of embeddings ({len(embeddings)}) and metadata entries ({len(metadata_entries)}).")
            return

        if self.index is None:
            self._initialize_index_if_needed(embeddings.shape[1])
        
        if self.index is None: 
            logger.error("Index not initialized. Cannot add embeddings.")
            return

        if embeddings.shape[1] != self.index.d:
            logger.error(f"Embedding dimension {embeddings.shape[1]} does not match index dimension {self.index.d}")
            
            return 

        self.index.add(embeddings.astype(np.float32))
        self.metadata_list.extend(metadata_entries)
        self.save_index()
        logger.info(f"Added {len(embeddings)} embeddings. Total vectors in index: {self.index.ntotal}")

   

    def search(self, query_embedding: np.ndarray, k: int = 5, document_id_filters: Optional[List[str]] = None) -> List[Tuple[str, Dict[str, Any], float]]:
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Search called on empty or uninitialized index.")
            return []
        
        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)
        
        if query_embedding.shape[1] != self.index.d:
            logger.error(f"Query embedding dimension {query_embedding.shape[1]} does not match index dimension {self.index.d}")
            return []

        
        k_to_retrieve = k * 5 if document_id_filters else k 
        k_to_retrieve = min(k_to_retrieve, self.index.ntotal)
        if k_to_retrieve == 0:
            k_to_retrieve = self.index.ntotal

        if k_to_retrieve == 0:
            return []

        distances, indices = self.index.search(query_embedding.astype(np.float32), k_to_retrieve)
        
        results = []
        if indices.size == 0 or distances.size == 0:
            return []
            
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx < 0 or idx >= len(self.metadata_list):
                continue

            meta_entry = self.metadata_list[idx]
            doc_id = meta_entry.get('doc_id', meta_entry.get('source_document', 'unknown_doc_id'))

            if document_id_filters and doc_id not in document_id_filters:
                continue
            
            results.append((doc_id, meta_entry, float(distances[0][i])))
            if len(results) == k:
                break
        
        logger.debug(f"Search returned {len(results)} results after filtering (if any). Initial k_to_retrieve: {k_to_retrieve}")
        return results


    def save_index(self):
        if self.index is not None:
            try:
                faiss.write_index(self.index, self.index_file_path)
                self._save_metadata_list()
                logger.info(f"Index and metadata saved to {self.index_path_base}. Total vectors: {self.index.ntotal}")
            except Exception as e:
                logger.error(f"Error saving index or metadata to {self.index_path_base}: {e}", exc_info=True)

    def _save_metadata_list(self):
        try:
            with open(self.meta_file_path, 'w') as f_meta:
                json.dump(self.metadata_list, f_meta, indent=2)
        except Exception as e:
            logger.error(f"Error saving metadata list to {self.meta_file_path}: {e}", exc_info=True)

    def load_index(self):
        try:
            logger.info(f"Loading index from {self.index_file_path}")
            self.index = faiss.read_index(self.index_file_path)
            logger.info(f"Loading metadata from {self.meta_file_path}")
            with open(self.meta_file_path, 'r') as f_meta:
                self.metadata_list = json.load(f_meta)
            
            if self.index.ntotal != len(self.metadata_list):
                logger.warning(f"Mismatch between FAISS index size ({self.index.ntotal}) and metadata list size ({len(self.metadata_list)}). Data may be inconsistent.")
            
            logger.info(f"Index and metadata loaded. Index size: {self.index.ntotal} vectors. Metadata entries: {len(self.metadata_list)}.")
        except FileNotFoundError:
            logger.error(f"Index or metadata file not found at {self.index_path_base}. Initializing as empty.")
            self.index = None
            self.metadata_list = []
        except Exception as e:
            logger.error(f"Error loading index or metadata from {self.index_path_base}: {e}", exc_info=True)
            self.index = None 
            self.metadata_list = []

    def get_index_size(self) -> int:
        return self.index.ntotal if self.index else 0

# Example usage (for testing)
# if __name__ == '__main__':
#     logging.basicConfig(level=logging.INFO)
#     sample_dim = 384 
#     test_index_path = "./test_data/test_faiss_store/my_index"
    
#     # Clean up old test files if they exist
#     if os.path.exists(test_index_path + ".index"): os.remove(test_index_path + ".index")
#     if os.path.exists(test_index_path + ".meta.json"): os.remove(test_index_path + ".meta.json")
#     if not os.path.exists(os.path.dirname(test_index_path)): os.makedirs(os.path.dirname(test_index_path))

#     vs_manager = VectorStoreManager(index_path=test_index_path, dimension=sample_dim)

#     if vs_manager.index is not None:
#         num_embeddings_doc1 = 3
#         dummy_embeddings_doc1 = np.random.rand(num_embeddings_doc1, sample_dim).astype(np.float32)
#         dummy_metadata_doc1 = [
#             {'doc_id': 'doc_1', 'chunk_text': f'Content from doc 1, chunk {i}', 'chunk_index': i} 
#             for i in range(num_embeddings_doc1)
#         ]
#         vs_manager.add_embeddings(dummy_embeddings_doc1, dummy_metadata_doc1)
#         logger.info(f"Added {num_embeddings_doc1} embeddings for doc_1.")

#         num_embeddings_doc2 = 2
#         dummy_embeddings_doc2 = np.random.rand(num_embeddings_doc2, sample_dim).astype(np.float32)
#         dummy_metadata_doc2 = [
#             {'doc_id': 'doc_2', 'chunk_text': f'Content from doc 2, chunk {i}', 'chunk_index': i} 
#             for i in range(num_embeddings_doc2)
#         ]
#         vs_manager.add_embeddings(dummy_embeddings_doc2, dummy_metadata_doc2)
#         logger.info(f"Added {num_embeddings_doc2} embeddings for doc_2.")

#         logger.info(f"Total vectors in index: {vs_manager.get_index_size()}")

#         query_vec = np.random.rand(1, sample_dim).astype(np.float32)
        
#         logger.info("--- Performing search (k=2, no filter) ---")
#         search_results_all = vs_manager.search(query_vec, k=2)
#         for doc_id, meta, dist in search_results_all:
#             logger.info(f"  Found: {doc_id}, chunk {meta.get('chunk_index')}, dist: {dist:.4f}, text: {meta['chunk_text'][:30]}...")

#         logger.info("--- Performing search (k=2, filter for doc_1) ---")
#         search_results_doc1 = vs_manager.search(query_vec, k=2, document_id_filters=['doc_1'])

#         for doc_id, meta, dist in search_results_doc1:
#             logger.info(f"  Found: {doc_id}, chunk {meta.get('chunk_index')}, dist: {dist:.4f}, text: {meta['chunk_text'][:30]}...")
#             assert doc_id == 'doc_1'
        
#         logger.info("--- Performing search (k=2, filter for doc_2) ---")
#         search_results_doc2 = vs_manager.search(query_vec, k=2, document_id_filters=['doc_2'])

#         for doc_id, meta, dist in search_results_doc2:
#             logger.info(f"  Found: {doc_id}, chunk {meta.get('chunk_index')}, dist: {dist:.4f}, text: {meta['chunk_text'][:30]}...")
#             assert doc_id == 'doc_2'

#         logger.info("--- Testing load from disk ---")
#         vs_manager_loaded = VectorStoreManager(index_path=test_index_path)
#         if vs_manager_loaded.index:
#             logger.info(f"Loaded index size: {vs_manager_loaded.get_index_size()}")
#             assert vs_manager_loaded.get_index_size() == (num_embeddings_doc1 + num_embeddings_doc2)
#             search_results_loaded = vs_manager_loaded.search(query_vec, k=1, document_id_filters=['doc_1'])  # ✅ correct

#             logger.info(f"Search on loaded index (doc_1, k=1): {len(search_results_loaded)} results")
#             assert len(search_results_loaded) > 0 if (num_embeddings_doc1 > 0) else True
#         else:
#             logger.error("Failed to load index for testing.")

#     else:
#         logger.error("VectorStoreManager could not be initialized with an index for testing.")
