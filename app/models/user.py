from sqlalchemy import Column, Integer, String, Boolean
from sqlalchemy.orm import relationship
from app.db.base_class import Base

class User(Base):
    __tablename__ = "user"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=True) # Nullable for Google OAuth users
    is_active = Column(Boolean(), default=True)
    is_superuser = Column(Boolean(), default=False)
    
    # This is the crucial link. It tells SQLAlchemy that a User can have many Chatbots.
    # The 'back_populates' argument must match the one in your Chatbot model.
    chatbots = relationship("Chatbot", back_populates="owner")