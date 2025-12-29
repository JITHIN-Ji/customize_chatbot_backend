from fastapi import Depends, HTTPException, Request
from jose import jwt, JWTError
from app.core.config import settings

def get_current_user_id(request: Request) -> str:
    """Extract the user ID ('sub') from our own JWT, stored in the Authorization header."""
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    token = auth_header.split(" ")[1]
    try:
        
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload["sub"]
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")