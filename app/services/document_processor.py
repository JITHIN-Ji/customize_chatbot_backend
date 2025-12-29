from typing import List, Dict, Tuple
import fitz  
import logging

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Processes documents by extracting text and chunking it.
    """

    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict]:
        pages: List[Dict] = []
        try:
            doc = fitz.open(pdf_path)
            for idx, page in enumerate(doc):
                text  = page.get_text() or ""
                # We get the page's specific label here, e.g., '1', 'ii', '6'
                page_label = page.get_label() or str(idx + 1)   
                pages.append({"index": idx, "label": page_label, "text": text})
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}", exc_info=True)
        return pages

    # --- THIS FUNCTION IS MODIFIED ---
    def chunk_text(
        self,
        pages: List[Dict],
        # Add new optional parameter to accept the filename or URL
        document_name: str = "source", 
        chunk_size: int = 1000,
        chunk_overlap: int = 100
    ) -> Tuple[List[str], List[Dict]]:
        chunks: List[str] = []
        meta: List[Dict] = []

        for page in pages:
            page_idx   = page["index"]
            original_page_label = page["label"] # This is the page number, e.g., '6'
            text       = page["text"]

            # --- NEW LOGIC: DETERMINE THE SOURCE LABEL ---
            # If a document_name was passed, use it. Otherwise, fall back to the page label.
            # This makes the function flexible.
            source_label = document_name if document_name != "source" else original_page_label

            start = 0
            while start < len(text):
                end   = start + chunk_size
                chunk = text[start:end]

                chunks.append(chunk)
                meta.append({
                    # Keep the original page number logic
                    "page": int(original_page_label) if original_page_label.isdigit() else page_idx + 1,
                    # --- NEW LOGIC: USE THE CORRECT, CONSISTENT LABEL ---
                    "label": source_label, 
                    "page_index": page_idx,
                    "chunk_index": len(chunks) - 1
                })

                start += chunk_size - chunk_overlap

        return chunks, meta


if __name__ == "__main__":
    proc = DocumentProcessor()
    # Test with the new parameter
    pages = proc.extract_text_from_pdf("example.pdf")
    # Pass the filename during the test
    ch, md = proc.chunk_text(pages, document_name="example.pdf")
    print(md[0])   
    # Expected output will now include the filename:
    # {'page': 1, 'label': 'example.pdf', 'page_index': 0, 'chunk_index': 0}