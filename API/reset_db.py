from sqlalchemy import inspect, text

from database import engine, Base
from models import Dataset, PatientRecord, Prediction, DatasetChunk, ChatSession, ChatMessage

print("Resetting database tables...")

expected_tables = {
    "datasets",
    "patient_records",
    "predictions",
    "dataset_chunks",
    "chat_sessions",
    "chat_messages",
}

inspector = inspect(engine)
existing_tables = set(inspector.get_table_names())

# Drop application tables tracked by the current metadata.
Base.metadata.drop_all(bind=engine)

# Remove legacy and unexpected tables left from the old auth-based design.
tables_to_remove = sorted(existing_tables - expected_tables)
if tables_to_remove:
    with engine.begin() as conn:
        for table_name in tables_to_remove:
            print(f"Dropping legacy table: {table_name}")
            conn.execute(text(f'DROP TABLE IF EXISTS "{table_name}" CASCADE'))

Base.metadata.create_all(bind=engine)
print("Database schema updated successfully!")
