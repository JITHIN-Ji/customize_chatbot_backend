import sys
import os

# Add the project's backend directory to the Python path FIRST
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI
from app.api import endpoints as api_endpoints  # Import the module
from app.services.agent import AgentService
from app.models.chat_models import ChatRequest, ChatResponse, DocumentUploadResponse
from unittest.mock import MagicMock, AsyncMock

# Create a new FastAPI app instance for testing
app = FastAPI()
app.include_router(api_endpoints.router)

# --- Test-specific Mocking ---
# Create a mock AgentService
mock_agent_service = MagicMock(spec=AgentService)

# Replace the global instance in the endpoints module with our mock
api_endpoints.agent_service_instance = mock_agent_service
# --- End Mocking ---

client = TestClient(app)

# Test data
TEST_PDF_FILE_NAME = "test.pdf"
TEST_TXT_FILE_NAME = "test.txt"
UPLOAD_DIR = "./test_uploads"

# Setup and teardown for tests
@pytest.fixture(autouse=True)
def setup_and_teardown():
    # Setup: Create dummy files and directories
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    with open(os.path.join(UPLOAD_DIR, TEST_PDF_FILE_NAME), "wb") as f:
        f.write(b"dummy pdf content")
    with open(os.path.join(UPLOAD_DIR, TEST_TXT_FILE_NAME), "w") as f:
        f.write("dummy txt content")
    
    # This is where the tests will run
    yield
    
    # Teardown: Clean up created files and directories
    # Reset mock after each test to clear call counts and configurations
    mock_agent_service.reset_mock()
    for file_name in os.listdir(UPLOAD_DIR):
        os.remove(os.path.join(UPLOAD_DIR, file_name))
    os.rmdir(UPLOAD_DIR)

def test_upload_document_success():
    """Test successful document upload."""
    mock_agent_service.handle_document_upload = AsyncMock(
        return_value=DocumentUploadResponse(
            document_id="test.pdf", 
            message="Upload successful",
            filename=TEST_PDF_FILE_NAME
        )
    )

    file_path = os.path.join(UPLOAD_DIR, TEST_PDF_FILE_NAME)
    with open(file_path, "rb") as f:
        response = client.post("/upload/", files={"file": (TEST_PDF_FILE_NAME, f, "application/pdf")})
    
    assert response.status_code == 200
    data = response.json()
    assert data["document_id"] == "test.pdf"
    assert data["message"] == "Upload successful"
    mock_agent_service.handle_document_upload.assert_awaited_once()


def test_upload_document_invalid_file_type():
    """Test uploading a file that is not a PDF."""
    file_path = os.path.join(UPLOAD_DIR, TEST_TXT_FILE_NAME)
    with open(file_path, "rb") as f:
        response = client.post("/upload/", files={"file": (TEST_TXT_FILE_NAME, f, "text/plain")})
    
    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]


def test_upload_document_no_file_name():
    """Test uploading a file with no name."""
    file_path = os.path.join(UPLOAD_DIR, TEST_PDF_FILE_NAME)
    with open(file_path, "rb") as f:
        response = client.post("/upload/", files={"file": ("", f, "application/pdf")})
    
    assert response.status_code == 422


def test_chat_with_agent_success():
    """Test successful chat interaction."""
    mock_response = ChatResponse(
        answer="This is a test answer.",
        sources=["doc1_chunk1", "doc1_chunk2"]
    )
    mock_agent_service.handle_chat_query = AsyncMock(return_value=mock_response)

    chat_request = ChatRequest(query="Hello?", chat_history=[])
    response = client.post("/chat/", json=chat_request.model_dump())

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "This is a test answer."
    assert len(data["sources"]) == 2
    assert data["sources"][0] == "doc1_chunk1"
    mock_agent_service.handle_chat_query.assert_awaited_once() 