from app.services.document_processor import DocumentProcessor
from app.services.embedding_generator import EmbeddingGenerator
from app.services.web_scraper import scrape_url
from app.core.config import settings
from app.services.llm_clients import GeminiClient, OpenAIClient
from typing import List, Dict, Any, Optional
import os
import logging
import numpy as np
from app.deps.vector import get_vs
import asyncio
import time
logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, user_id: str):
        
        self.gemini_client = GeminiClient() if settings.GEMINI_API_KEY else None
        self.openai_client = OpenAIClient() if settings.OPENROUTER_API_KEY else None
        self.doc_processor = DocumentProcessor()
        self.embed_generator = EmbeddingGenerator(gemini_client=self.gemini_client)
        
        self.vector_store = get_vs(user_id) 
        logger.info("RAGPipeline initialized.")

     

    async def process_and_embed_document(self, file_path: str, document_id: str) -> tuple[bool, str]:
        logger.info(f"Processing document: {file_path} with ID: {document_id}")
        
        
        start = time.time()
        pages = self.doc_processor.extract_text_from_pdf(file_path)
        logger.info(f"Text extraction took {time.time() - start:.2f} seconds")

        if not pages:
            message = f"No text extracted from {file_path}"
            logger.warning(message)
            return False, message

        
        start = time.time()
        # The new, corrected line
        chunks, meta_per_chunk = self.doc_processor.chunk_text(pages, document_name=document_id)
        logger.info(f"Chunking took {time.time() - start:.2f} seconds")

        if not chunks:
            message = f"No chunks created for {file_path}"
            logger.warning(message)
            return False, message

        logger.info(f"Generated {len(chunks)} chunks for document {document_id}.")

        
        start = time.time()
        BATCH_SIZE = 64 
        async def embed_all_batches(chunks, batch_size):
            tasks = []
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                logger.info(f"Embedding batch {i} to {i + len(batch)}...")
                tasks.append(self.embed_generator.generate_embeddings(batch, task_type="RETRIEVAL_DOCUMENT"))
            results = await asyncio.gather(*tasks)
            return [embedding for batch in results if batch is not None for embedding in batch]

        
        all_embeddings = await embed_all_batches(chunks, BATCH_SIZE)

        embeddings = np.array(all_embeddings, dtype=np.float32)
        logger.info(f"Embedding generation for all chunks took {time.time() - start:.2f} seconds")

        if embeddings is None or embeddings.size == 0:
            message = f"Failed to generate embeddings for {document_id}"
            logger.error(message)
            return False, message

    
        self.vector_store.add_embeddings(
            embeddings,
            [
                {
                    'doc_id': document_id,
                    'chunk_text': chunk,
                    'chunk_index': i,
                    
                    'page': meta.get('page'),
                    'label': meta.get('label')  
                }
                for i, (chunk, meta) in enumerate(zip(chunks, meta_per_chunk))
            ]
        )

        message = f"Embeddings for {document_id} added to vector store."
        logger.info(message)
        return True, message
    



    



    async def process_and_embed_web_content(self, url: str, document_id: str) -> tuple[bool, str]:
        """
        Scrapes a URL, processes the text, and stores its embeddings.
        """
        logger.info(f"Processing web content from URL: {url} with ID: {document_id}")

        try:
            # Step 1: Scrape the text content from the URL
            scraped_text = scrape_url(url)
            if not scraped_text:
                message = f"No text content could be scraped from {url}"
                logger.warning(message)
                return False, message
        except Exception as e:
            logger.error(f"Failed to scrape URL {url}: {e}")
            return False, str(e)

        # Step 2: Chunk the scraped text (treating the page as a single document)
        web_page_content = [{"index": 0, "label": url, "text": scraped_text}]
        chunks, meta_per_chunk = self.doc_processor.chunk_text(web_page_content)
        if not chunks:
            message = f"No chunks created for URL {url}"
            logger.warning(message)
            return False, message
        logger.info(f"Generated {len(chunks)} chunks for URL {document_id}.")

        # Step 3: Generate embeddings for the chunks (reusing existing async logic)
        BATCH_SIZE = 64
        async def embed_all_batches(chunks, batch_size):
            tasks = []
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                tasks.append(self.embed_generator.generate_embeddings(batch, task_type="RETRIEVAL_DOCUMENT"))
            results = await asyncio.gather(*tasks)
            return [embedding for batch in results if batch is not None for embedding in batch]

        all_embeddings = await embed_all_batches(chunks, BATCH_SIZE)
        embeddings = np.array(all_embeddings, dtype=np.float32)

        if embeddings is None or embeddings.size == 0:
            message = f"Failed to generate embeddings for {document_id}"
            logger.error(message)
            return False, message

        # Step 4: Add embeddings to the vector store with URL as a source label
        self.vector_store.add_embeddings(
            embeddings,
            [
                {
                    'doc_id': document_id,
                    'chunk_text': chunk,
                    'chunk_index': i,
                    'page': 1,  # We can consider a webpage as a single page
                    'label': url,  # Use the URL as the label for source tracking
                }
                for i, chunk in enumerate(chunks)
            ]
        )

        message = f"Embeddings for URL {document_id} added to vector store."
        logger.info(message)
        return True, message
    
    async def process_and_embed_document_from_chunks(self, chunks: List[str], meta_per_chunk: List[dict], document_id: str) -> tuple[bool, str]:
        """
        Used when OCR text (from image) is already available and chunked.
        """
        logger.info(f"Embedding OCR-based chunks for document {document_id}")
        BATCH_SIZE = 64

        async def embed_all_batches(chunks, batch_size):
            tasks = []
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                tasks.append(self.embed_generator.generate_embeddings(batch, task_type="RETRIEVAL_DOCUMENT"))
            results = await asyncio.gather(*tasks)
            return [embedding for batch in results if batch is not None for embedding in batch]

        all_embeddings = await embed_all_batches(chunks, BATCH_SIZE)
        embeddings = np.array(all_embeddings, dtype=np.float32)

        if embeddings is None or embeddings.size == 0:
            return False, f"Failed to embed OCR chunks for {document_id}"

        self.vector_store.add_embeddings(
            embeddings,
            [
                {
                    "doc_id": document_id,
                    "chunk_text": chunk,
                    "chunk_index": i,
                    "page": meta_per_chunk[i].get("page", 1), # Images are usually one page
                    "label": document_id # Use the document_id (filename) as the label
                }
                for i, chunk in enumerate(chunks)
            ]
        )

        return True, f"OCR document {document_id} embedded successfully."



    async def retrieve_relevant_chunks(self, query: str, k: int = settings.TOP_K, document_id_filters: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Retrieves relevant text chunks for a given query, optionally filtered by document_ids."""
        if self.vector_store.index is None or self.vector_store.index.ntotal == 0:
            logger.warning("Vector store is empty or not initialized. Cannot retrieve.")
            return []
             
        query_embedding = await self.embed_generator.generate_embeddings([query], task_type="RETRIEVAL_QUERY")
        if query_embedding is None or query_embedding.size == 0:
            logger.error("Failed to generate query embedding.")
            return []
        
        results = self.vector_store.search(query_embedding, k=k, document_id_filters=document_id_filters)
        
        retrieved_chunks = []
        for res_doc_id, meta, distance in results:
            if meta:
                chunk_info = {
                    'document_id': meta.get('doc_id', res_doc_id),  
                    'text': meta.get('chunk_text', ''),
                    'chunk_index': meta.get('chunk_index', -1),
                    'page': meta.get('page', -1), 
                    'label': meta.get('label', 'N/A'), 
                }
                retrieved_chunks.append(chunk_info)
        
        logger.info(f"Retrieved {len(retrieved_chunks)} chunks for query.")
        return retrieved_chunks

    async def generate_answer(
        self, 
        query: str, 
        context_chunks: List[Dict[str, Any]], 
        llm_provider: str, 
        chat_history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None # New parameter
    ):
        """Generates an answer using an LLM based on query, context, and chat history."""
        context_str = "\n".join([
            f"Document: {chunk['document_id']} (Page {chunk.get('page', 'N/A')}):\n{chunk['text']}"
            for chunk in context_chunks
        ])

        if context_str:
            
            persona_injection = ""
            if system_prompt:
                persona_injection = f"""
**Your Persona:**
You must adopt the following personality and instructions for your response: "{system_prompt}"
"""
            
            prompt = f"""You are a helpful and thorough AI document assistant.{persona_injection}

**Critical Rules:**
1.  **NEVER use outside knowledge.** Your knowledge is strictly limited to the text in the "Provided Document Context".
2.  **Provide a Detailed and Comprehensive Answer:** If the context contains the information, synthesize a thorough answer. Explain the key points in detail and structure your response in a clear, logical way. Use bullet points or numbered lists if it helps improve clarity. Do not be overly concise.
3.  **If the context does NOT contain the answer...** (This rule remains the same)

**Provided Document Context:**
---
{context_str}
---

**User's Question:** {query}

**Your Detailed Answer:**
"""
        else:
            prompt = f"""You are 'Cascade', a friendly AI document assistant. A user has asked a question, but no relevant information could be found in their uploaded documents.

**Rules:**
1.  Do NOT use your general knowledge.
2.  Politely inform the user that the information is not in the documents.

**User's Question:** {query}

**Your Answer:**
"""

        logger.debug(f"--- Prompt for LLM ({llm_provider}) ---\nSystem Prompt: {system_prompt}\nContext length: {len(context_str)}\nPrompt: {prompt[:500]}...\n-----------------------")

        llm_response = "No LLM provider was selected or available."
        
        if llm_provider == "gemini":
            if self.gemini_client:
                llm_response = await self.gemini_client.generate_text(prompt, history=chat_history)
            else:
                llm_response = "Gemini client is not available (e.g., API key missing)."
        elif llm_provider == "openai":
            if self.openai_client:
                llm_response = await self.openai_client.generate_text(prompt, history=chat_history)
            else:
                llm_response = "OpenAI client is not available (e.g., API key missing)."
        
        return llm_response