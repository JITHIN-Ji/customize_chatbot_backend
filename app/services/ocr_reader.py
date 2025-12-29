import os
import logging
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.exceptions import ClientAuthenticationError, HttpResponseError

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

# Get Document Intelligence endpoint and key from environment variables
doc_intelligence_endpoint = os.getenv('DOC_INTELLIGENCE_ENDPOINT')
doc_intelligence_key = os.getenv('DOC_INTELLIGENCE_KEY')

class OCRReader:
    """
    A wrapper around the Azure AI Document Intelligence "read" model for OCR.
    """

    def __init__(self):
        """
        Initializes the DocIntellOCRReader by creating a DocumentAnalysisClient.
        Authentication is handled via the DOC_INTELLIGENCE_ENDPOINT and 
        DOC_INTELLIGENCE_KEY environment variables.
        """
        self.client = None
        if not doc_intelligence_endpoint or not doc_intelligence_key:
            logger.error(
                "Document Intelligence endpoint or key not found. "
                "Please set DOC_INTELLIGENCE_ENDPOINT and DOC_INTELLIGENCE_KEY."
            )
            return

        try:
            self.client = DocumentAnalysisClient(
                endpoint=doc_intelligence_endpoint,
                credential=AzureKeyCredential(doc_intelligence_key)
            )
        except ClientAuthenticationError:
            logger.error(
                "Authentication failed. Please check your Document Intelligence endpoint and key."
            )
        except Exception as e:
            logger.error(f"Failed to initialize Document Intelligence client: {e}")

    def extract_text(self, image_path: str) -> str:
        """
        Detects and extracts text from an image file using the 
        Azure AI Document Intelligence "read" model.

        Args:
            image_path: The path to the local image file.

        Returns:
            The extracted text as a single string, or an empty string if OCR fails.
        """
        if not self.client:
            logger.error("Document Intelligence client is not initialized. Cannot perform OCR.")
            return ""

        try:
            with open(image_path, "rb") as f:
                image_data = f.read()

            # Use the "prebuilt-read" model for general OCR
            poller = self.client.begin_analyze_document(
                model_id="prebuilt-read", 
                document=image_data
            )
            
            # Wait for the analysis to complete and get the result
            result = poller.result()

            # The full extracted text is available in the content attribute
            if result.content:
                return result.content
            else:
                return ""

        except HttpResponseError as e:
            logger.error(f"Document Intelligence API call failed for {image_path}: {e.message}")
            return ""
        except Exception as e:
            logger.error(f"An unexpected error occurred during OCR for {image_path}: {e}", exc_info=True)
            return ""

# Example of how to use the new class:
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Create an instance of the reader
    doc_intell_reader = DocIntellOCRReader()

    # Make sure the client was initialized
    if doc_intell_reader.client:
        # Provide the path to your image
        # To run this example, you must have an image named 'sample.jpg' 
        # in the same directory, or change the path to your image file.
        image_to_test = "sample.jpg" 

        if os.path.exists(image_to_test):
            # Extract the text
            extracted_text = doc_intell_reader.extract_text(image_to_test)

            # Print the result
            if extracted_text:
                print("--- Extracted Text (Document Intelligence) ---")
                print(extracted_text)
                print("---------------------------------------------")
            else:
                print("No text was extracted from the image.")
        else:
            print(f"Error: The image file '{image_to_test}' was not found.")
            print("Please create a 'sample.jpg' file or update the 'image_to_test' variable with the correct path.")