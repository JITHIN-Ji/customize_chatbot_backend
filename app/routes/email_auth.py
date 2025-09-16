from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session # 1. IMPORT Session for database connection
from datetime import timedelta     # 2. IMPORT timedelta for token expiration

# --- 3. IMPORT from your new, correct modules ---
from app.models import chat_models as schemas
from app.models.user import User                  # The SQLAlchemy User model
from app.db.session import get_db                 # The database session dependency
from app import security                                   # Your top-level security.py file
from app.core.config import settings              # Your main settings

router = APIRouter()

@router.post("/register", response_model=schemas.UserInDB, tags=["Email Authentication"])
def register_user(
    *,
    db: Session = Depends(get_db), # 4. ADD the database dependency here
    user_in: schemas.UserCreate
):
    """
    Handles user registration and saves them to the main SQL database.
    """
    # 5. Check for the user in the main database using SQLAlchemy
    db_user = db.query(User).filter(User.email == user_in.email).first()
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )
    
    # 6. Create a new User object for the main database
    hashed_password = security.get_password_hash(user_in.password)
    new_user = User(
        email=user_in.email, 
        hashed_password=hashed_password
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user


@router.post("/token", response_model=schemas.Token, tags=["Email Authentication"])
def login_for_access_token(
    db: Session = Depends(get_db), # 7. ADD the database dependency here
    form_data: OAuth2PasswordRequestForm = Depends()
):
    """
    Handles user login against the main SQL database and returns a JWT.
    """
    # 8. Find the user in the main database using SQLAlchemy
    user = db.query(User).filter(User.email == form_data.username).first()
    
    # 9. Verify the password against the user record from the database
    if not user or not user.hashed_password or not security.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # 10. Create the access token using the function from your security.py file
    # This correctly sets the expiration time to 24 hours.
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = security.create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}