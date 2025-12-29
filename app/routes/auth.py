import os
import time
# --- 1. IMPORT datetime and timedelta ---
from datetime import datetime, timedelta 
from fastapi import APIRouter, Request, HTTPException, Depends
from starlette.responses import RedirectResponse
from authlib.integrations.starlette_client import OAuth
from jose import jwt
from sqlalchemy.orm import Session

from app.core.config import settings
from . import email_auth
from app.db.session import get_db
from app.models.user import User

router = APIRouter(prefix="/auth", tags=["Authentication"])

oauth = OAuth()
oauth.register(
    name="google",
    client_id=settings.GOOGLE_CLIENT_ID,
    client_secret=settings.GOOGLE_CLIENT_SECRET,
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={
        "scope": "openid email profile",
        "response_type": "code",
    },
)

router.include_router(email_auth.router)


@router.get("/login", tags=["Google Authentication"])
async def login(request: Request):
    redirect_uri = request.url_for("auth_callback")
    return await oauth.google.authorize_redirect(request, redirect_uri)


@router.get("/callback", name="auth_callback", tags=["Google Authentication"])
async def auth_callback(
    request: Request,
    db: Session = Depends(get_db)
):
    try:
        token = await oauth.google.authorize_access_token(request)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not authorize access token: {e}")

    id_token = token.get("id_token")
    if not id_token:
        raise HTTPException(status_code=400, detail="Missing ID token in Google response")

    try:
        user_info = jwt.decode(
            id_token,
            key=None,
            algorithms=["RS256"],
            options={
                "verify_signature": False,
                "verify_at_hash": False
            },
            audience=settings.GOOGLE_CLIENT_ID
        )
    except jwt.JWTClaimsError as e:
        raise HTTPException(status_code=400, detail=f"Invalid token claims: {e}")

    if not user_info:
        raise HTTPException(status_code=400, detail="Google login failed")

    email = user_info.get("email")
    if not email:
        raise HTTPException(status_code=400, detail="Email not found in Google token")

    db_user = db.query(User).filter(User.email == email).first()

    if not db_user:
        print(f"User with email {email} not found. Creating new user.")
        new_user_data = User(email=email)
        db.add(new_user_data)
        db.commit()
        db.refresh(new_user_data)
        db_user = new_user_data
    else:
        print(f"Found existing user with email {email}. Logging them in.")

    # --- 2. THIS IS THE FIX ---
    # Create the expiration time using the setting from your config.py file.
    # This will create a token that lasts for 24 hours.
    expire = datetime.utcnow() + timedelta(
        minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
    )
    
    # The data to encode in the token, now with the correct expiration time.
    to_encode = {"sub": db_user.email, "exp": expire}
    
    # The old, incorrect line has been replaced with the 'to_encode' dictionary.
    app_token = jwt.encode(
        to_encode,
        settings.SECRET_KEY, 
        algorithm=settings.ALGORITHM, 
    )
    
    return RedirectResponse(url=f"http://localhost:3000/?token={app_token}")