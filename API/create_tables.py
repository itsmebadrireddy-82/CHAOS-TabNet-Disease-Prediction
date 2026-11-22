from database import engine, Base
from models import Dataset, PatientRecord, Prediction, DatasetChunk, ChatSession, ChatMessage

print("Creating database tables...")
Base.metadata.create_all(bind=engine)
print("Tables created successfully!")
