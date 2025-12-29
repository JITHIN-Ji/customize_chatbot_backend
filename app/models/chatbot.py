from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship
from app.db.base_class import Base # Assuming you have this for SQLAlchemy

class Chatbot(Base):
    __tablename__ = "chatbots"

    id = Column(Integer, primary_key=True, index=True)
    chatbot_title = Column(String, index=True)
    welcome_message = Column(String)
    system_prompt = Column(String) # This is where chatbotInstructions will be stored
    avatar_url = Column(String, nullable=True)
    user_icon_url = Column(String, nullable=True)
    bubble_icon_url = Column(String, nullable=True) # <-- ADD THIS LINE
    document_id = Column(String, nullable=True) # To link to the trained document in the vector store
    
    owner_id = Column(Integer, ForeignKey("user.id"))
    owner = relationship("User", back_populates="chatbots")
