from fastapi import APIRouter, File, UploadFile, HTTPException
import os
from datetime import datetime
import openai
from openai import AsyncOpenAI
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# --- OpenAI API Configuration ---
# The API key is read from the OPENAI_API_KEY environment variable.
try:
    client = AsyncOpenAI()
    logger.info("OpenAI client initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {e}")
    # This will prevent the app from starting if the key is not set, which is good practice.
    raise ValueError("OPENAI_API_KEY environment variable not set or invalid.") from e


@router.post("/")
async def upload_audio_for_transcription(audio: UploadFile = File(...)):
    """
    Transcribes audio using OpenAI Whisper with automatic language detection.
    """
    logger.info(f"Received audio file: {audio.filename}, content_type: {audio.content_type}")

    # Validate file type
    allowed_formats = ['.mp3', '.mp4', '.mpeg', '.mpga', '.m4a', '.wav', '.webm']
    file_ext = os.path.splitext(audio.filename)[1].lower()
    
    if file_ext not in allowed_formats:
        logger.warning(f"Unsupported file extension: {file_ext}")
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported audio format. Supported formats: {', '.join(allowed_formats)}"
        )
    
    # Read audio content directly into memory
    audio_content = await audio.read()
    
    if not audio_content:
        raise HTTPException(status_code=400, detail="Empty audio file provided.")
        
    try:
        logger.info("Sending audio to OpenAI Whisper for transcription...")
        
        # The OpenAI library requires a file-like object, which is a tuple of
        # (filename, file_content, content_type)
        audio_file = (audio.filename, audio_content, audio.content_type)
        
        transcript = await client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
        
        logger.info(f"Transcription successful. Transcript length: {len(transcript.text)}")

    except openai.APIError as e:
        logger.error(f"OpenAI API Error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to transcribe audio due to an API error: {e.body.get('message', str(e))}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during transcription: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

    return {
        "message": "Audio transcribed successfully",
        "filename": audio.filename,
        "transcript": transcript.text,
        "language": "auto-detected"  # Whisper auto-detects but doesn't return the language code in the response
    }


@router.post("/with-language")
async def upload_audio_with_language(
    audio: UploadFile = File(...),
    language: str = "en"  # Default to English. Use ISO-639-1 code.
):
    """
    Transcribes audio using OpenAI Whisper with a specified language.
    """
    logger.info(f"Received audio for transcription with specified language: {language}")
    
    # A list of some common ISO-639-1 language codes supported by Whisper.
    # For a full list, refer to OpenAI's documentation.
    supported_languages = [
        "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca",
        "nl", "ar", "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms",
        "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la"
    ]
    
    if language not in supported_languages:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported language code: '{language}'. Please use a valid ISO-639-1 code."
        )

    audio_content = await audio.read()
    if not audio_content:
        raise HTTPException(status_code=400, detail="Empty audio file provided.")

    try:
        audio_file = (audio.filename, audio_content, audio.content_type)
        
        transcript = await client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language=language  # Pass the specified language to the API
        )
        
        logger.info("Transcription successful with specified language.")

    except openai.APIError as e:
        logger.error(f"OpenAI API Error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to transcribe audio: {e.body.get('message', str(e))}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    
    return {
        "message": "Audio transcribed successfully",
        "filename": audio.filename,
        "transcript": transcript.text,
        "language": language
    }