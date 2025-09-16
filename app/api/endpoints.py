from app.models.chat_models import ChatbotCreate, Chatbot as ChatbotSchema
import uuid
import aiofiles
from typing import Annotated, List
from app.models.chatbot import Chatbot as ChatbotModel
from app.models.user import User
from app.db.session import get_db
from sqlalchemy.orm import Session
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from app.models.chat_models import ChatRequest, ChatResponse, DocumentUploadResponse
from app.services.agent import AgentService
from app.deps.auth import get_current_user_id
from app.services.ocr_reader import OCRReader
from app.models.chat_models import ChatRequest, ChatResponse, DocumentUploadResponse, URLIngestRequest
from app.api.audio_routes import router as audio_router
from app.core.config import settings
import os, shutil
from app.api import deps
from app.models.chat_models import ChatbotPublic as ChatbotPublicSchema

router = APIRouter()
router.include_router(audio_router, prefix="/upload-audio", tags=["Audio"])

ocr_reader = OCRReader()

# --- 2. NEW: DEFINE CONSTANTS FOR THE ICON UPLOAD LOGIC ---
# This makes your code cleaner and easier to manage.
AVATAR_UPLOAD_DIR = "uploads/avatars"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}
MAX_FILE_SIZE_MB = 5
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


async def get_agent_service(user_id: str = Depends(get_current_user_id)):
    return AgentService(user_id)



@router.post("/upload/", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user_id), 
    agent_service: AgentService = Depends(get_agent_service),
):
    # ... your existing code ...
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file name provided.")
    
    ext = file.filename.lower().split(".")[-1]
    if ext not in ["pdf", "png", "jpg", "jpeg"]:
        raise HTTPException(status_code=400, detail="Only PDF or image files are allowed.")

    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(settings.UPLOAD_DIR, file.filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save file: {e}")
    finally:
        file.file.close()
    
    if ext == "pdf":
        result = await agent_service.handle_document_upload(file_path, file.filename)
    else:
        text = ocr_reader.extract_text(file_path)
        result = await agent_service.handle_image_upload(text, file.filename)

    return result

@router.post("/ingest-url/", response_model=DocumentUploadResponse)
async def ingest_url(
    request: URLIngestRequest,
    user_id: str = Depends(get_current_user_id),
    agent_service: AgentService = Depends(get_agent_service),
):
    # ... your existing code ...
    if not request.url:
        raise HTTPException(status_code=400, detail="No URL provided.")
    
    result = await agent_service.handle_url_ingestion(str(request.url))
    
    return result

@router.post("/chat/", response_model=ChatResponse)
async def chat_with_agent(
    request: ChatRequest,
    db: Session = Depends(get_db),
    # 4. Use the imported dependency
    current_user: User = Depends(deps.get_current_user),      
    agent_service: AgentService = Depends(get_agent_service)
):
    chatbot = db.query(ChatbotModel).filter(ChatbotModel.id == request.chatbot_id).first()
    if not chatbot or chatbot.owner_id != current_user.id:
         raise HTTPException(status_code=404, detail="Chatbot not found or you do not have permission to access it.")

    document_ids_to_use = [chatbot.document_id] if chatbot.document_id else []
    system_prompt_to_use = chatbot.system_prompt
    
    response = await agent_service.handle_chat_query(
        query=request.query,
        chat_history=request.chat_history,
        document_ids=document_ids_to_use,
        system_prompt=system_prompt_to_use,
        user_id=current_user.id,
        language=request.language
    )   
    return response

@router.post("/upload-icon", tags=["Settings"])
async def upload_icon(
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user_id) 
):
    """
    Handles uploading a user or bot icon.
    """
    file_extension = file.filename.split(".")[-1].lower()
    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File is too large. Maximum size is {MAX_FILE_SIZE_MB}MB."
        )

    unique_filename = f"{uuid.uuid4()}.{file_extension}"
    # Use the constant for the directory
    os.makedirs(AVATAR_UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(AVATAR_UPLOAD_DIR, unique_filename)

    try:
        async with aiofiles.open(file_path, 'wb') as out_file:
            await out_file.write(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"There was an error saving the file: {e}")

    file_url = f"/{AVATAR_UPLOAD_DIR}/{unique_filename}"
    
    return {"message": "Icon uploaded successfully", "url": file_url}

@router.post("/chatbots/", response_model=ChatbotSchema, tags=["Chatbots"])
def create_chatbot(
    *,
    db: Session = Depends(get_db),
    chatbot_in: ChatbotCreate,
    # 2. Use the imported dependency
    current_user: User = Depends(deps.get_current_user) 
):
    """Creates a new chatbot record in the database."""
    db_chatbot = ChatbotModel(
        chatbot_title=chatbot_in.chatbotTitle,
        welcome_message=chatbot_in.welcomeMessage,
        system_prompt=chatbot_in.chatbotInstructions,
        avatar_url=chatbot_in.avatarUrl,
        user_icon_url=chatbot_in.userIconUrl,
        bubble_icon_url=chatbot_in.bubbleIconUrl, # <-- ADD THIS LINE
        document_id=chatbot_in.documentId,
        owner_id=current_user.id
    )
    db.add(db_chatbot)
    db.commit()
    db.refresh(db_chatbot)
    return db_chatbot

# @router.get("/chatbots/", response_model=List[ChatbotSchema], tags=["Chatbots"])
# def read_my_chatbots(
#     db: Session = Depends(get_db),
#     # 3. Use the imported dependency
#     current_user: User = Depends(deps.get_current_user)
# ):
#     """Retrieves all chatbots owned by the current user."""
#     return db.query(ChatbotModel).filter(ChatbotModel.owner_id == current_user.id).all()

@router.get("/chatbots/{chatbot_id}/public", response_model=ChatbotPublicSchema, tags=["Public"])
def get_public_chatbot_config(
    chatbot_id: int,
    db: Session = Depends(get_db)
):
    """
    Retrieve the public-facing configuration for a specific chatbot.
    This endpoint is NOT authenticated and is safe to be called from any website.
    """
    chatbot = db.query(ChatbotModel).filter(ChatbotModel.id == chatbot_id).first()
    
    if not chatbot:
        # If no chatbot with that ID is found, return a 404 error.
        raise HTTPException(status_code=404, detail="Chatbot not found")
        
    # Pydantic will automatically convert your ChatbotModel object
    # into the secure ChatbotPublicSchema, which only includes public data.
    return chatbot

@router.post("/chat/public", response_model=ChatResponse, tags=["Public"])
async def public_chat_with_agent(
    request: ChatRequest,
    db: Session = Depends(get_db)
):
    """
    Handles a chat request from a public, embedded chatbot.
    """
    # 1. Find the chatbot in the database.
    chatbot = db.query(ChatbotModel).filter(ChatbotModel.id == request.chatbot_id).first()
    
    if not chatbot:
         raise HTTPException(status_code=404, detail="Chatbot not found.")

    # --- THIS IS THE CRITICAL FIX ---
    # 2. Get the OWNER's EMAIL from the chatbot's owner relationship.
    #    SQLAlchemy automatically loads the related 'owner' object from the User table.
    owner_email = chatbot.owner.email
    
    # 3. Initialize the AgentService with the CORRECT owner's EMAIL.
    #    This ensures it looks in the correct folder (e.g., .../jithinjithuedpl922@gmail.com/...)
    public_agent_service = AgentService(user_id=owner_email)

    # 4. Get the saved settings from the chatbot record.
    document_ids_to_use = [chatbot.document_id] if chatbot.document_id else []
    system_prompt_to_use = chatbot.system_prompt
    
    # 5. Call the agent's brain, passing the saved settings.
    response = await public_agent_service.handle_chat_query(
        query=request.query,
        chat_history=request.chat_history,
        document_ids=document_ids_to_use,
        system_prompt=system_prompt_to_use,
        # The user_id for saving public chat history can still be generic
        user_id=f"public_user_for_chatbot_{request.chatbot_id}",
        language=request.language
    )   
    return response