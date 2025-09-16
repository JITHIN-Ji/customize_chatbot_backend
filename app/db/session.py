from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

    # This line tells the app to create a database file named "chatbots.db"
SQLALCHEMY_DATABASE_URL = "sqlite:///./chatbots.db"

engine = create_engine(
        SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()