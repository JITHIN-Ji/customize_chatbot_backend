import google.generativeai as genai
# from groq import Groq, AsyncGroq
from app.core.config import settings
from typing import List, Dict, Any, Optional
import asyncio
import logging
import openai
logger = logging.getLogger(__name__)

class GeminiClient:
    def __init__(self):
        if not settings.GEMINI_API_KEY:
            logger.warning("Gemini API Key not found. GeminiClient will not be functional.")
            self.gen_model = None
            self.emb_model_name = None
            return
        
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.gen_model = genai.GenerativeModel(settings.GEMINI_MODEL_NAME)
        self.emb_model_name = settings.GEMINI_EMBEDDING_MODEL_NAME
        logger.info(f"GeminiClient initialized with generation model: {settings.GEMINI_MODEL_NAME} and embedding model: {self.emb_model_name}")

    async def generate_text(self, prompt: str, history: Optional[List[Any]] = None) -> str:
        if not self.gen_model:
            return "Gemini client not configured due to missing API key."
        
        gemini_history = []
        if history:
            for item in history:
                role = 'user' if item.get('role') == 'user' else 'model'
                gemini_history.append({'role': role, 'parts': [item.get('content', '')]})
        
        try:
            chat_session = self.gen_model.start_chat(history=gemini_history if gemini_history else None)
            response = await chat_session.send_message_async(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error generating text with Gemini: {e}")
            return f"Error from Gemini: {str(e)}"


class OpenAIClient:
    def __init__(self):
        if not settings.OPENROUTER_API_KEY:
            logger.warning("OpenRouter API Key (OPENROUTER_API_KEY) not found. Client will not be functional.")
            self.client = None
            self.model_name = None
            return

        # Configure the client to point to OpenRouter's API endpoint.
        self.client = openai.AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.OPENROUTER_API_KEY,
        )

        # Prepare optional headers for OpenRouter analytics
        self.default_headers = {}
        if settings.OPENROUTER_SITE_URL:
            self.default_headers["HTTP-Referer"] = settings.OPENROUTER_SITE_URL
        if settings.OPENROUTER_SITE_NAME:
            self.default_headers["X-Title"] = settings.OPENROUTER_SITE_NAME
        
        # The model name should be in OpenRouter format, e.g., "openai/gpt-4o-mini"
        self.model_name = settings.OPENAI_MODEL
        logger.info(f"OpenAIClient initialized for OpenRouter with model: {self.model_name}")

    async def generate_text(self, prompt: str, history: Optional[List[Any]] = None) -> str:
        if not self.client:
            return "OpenRouter client is not available (e.g., API key missing)."
        
        messages = []
        if history:
            for item in history:
                role = 'user' if item.get('role') == 'user' else 'assistant'
                content = item.get('content', '') 
                messages.append({"role": role, "content": content})
                
        messages.append({"role": "user", "content": prompt})

        try:
            # Pass the default headers with every request
            chat_completion = await self.client.chat.completions.create(
                extra_headers=self.default_headers, # Pass the headers here
                messages=messages,
                model=self.model_name,
                temperature=0.1,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating text with OpenRouter: {e}")
            return f"Error from OpenRouter: {str(e)}"

# class GroqClient:
#     def __init__(self):
#         if not settings.GROQ_API_KEY:
#             logger.warning("Groq API Key not found. GroqClient will not be functional.")
#             self.client = None
#             self.model_name = None
#             return
# 
#         self.client = AsyncGroq(api_key=settings.GROQ_API_KEY)
#         self.model_name = settings.GROQ_MODEL_NAME
#         logger.info(f"GroqClient initialized with model: {self.model_name}")
# 
#     async def generate_text(self, prompt: str, history: Optional[List[Any]] = None) -> str:
#         if not self.client:
#             return "Groq client not configured due to missing API key."
#         
#         
#         messages = []
#         if history:
#             for item in history:
#                 
#                 
#                 role = 'user' if item.get('role') == 'user' else 'assistant'
#                 content = item.get('content', '') 
#                 messages.append({"role": role, "content": content})
#                 
#         messages.append({"role": "user", "content": prompt})
#         try:
#             chat_completion = await self.client.chat.completions.create(
#                 messages=messages,
#                 model=self.model_name,
#             )
#             return chat_completion.choices[0].message.content
#         except Exception as e:
#             logger.error(f"Error generating text with Groq: {e}")
#             return f"Error from Groq: {str(e)}"
# --- MODIFIED AREA END ---