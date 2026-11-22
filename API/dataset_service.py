import pandas as pd
import json
from sqlalchemy.orm import Session
from models import Dataset, DatasetChunk, PatientRecord


TARGET_COLUMN_CANDIDATES = (
    "diagnosis",
    "target",
    "label",
    "result",
    "outcome",
    "class",
    "prediction",
    "status",
    "has_disease",
    "disease",
)


def _normalize_column_name(value: str | None) -> str:
    if not value:
        return ""
    return "".join(ch.lower() for ch in value if ch.isalnum())


def _guess_target_column(df: pd.DataFrame) -> str | None:
    columns = list(df.columns)
    if not columns:
        return None

    normalized_to_actual = {_normalize_column_name(str(col)): str(col) for col in columns}
    for candidate in TARGET_COLUMN_CANDIDATES:
        actual = normalized_to_actual.get(_normalize_column_name(candidate))
        if actual:
            return actual

    binary_columns: list[str] = []
    for col in columns:
        non_null = df[col].dropna()
        if non_null.empty:
            continue
        unique_count = non_null.astype(str).nunique()
        if 1 < unique_count <= 2:
            binary_columns.append(str(col))

    if binary_columns:
        return binary_columns[0]

    return str(columns[-1])

def process_uploaded_dataset(dataset_id: int, file_path: str, df: pd.DataFrame, db: Session):
    """
    1. Extracts metadata (potential features).
    2. Chunks the data for "RAG-like" context or just batching.
    3. Updates the Dataset record.
    """
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        return
    
    # 1. Metadata Extraction
    # Heuristic: Columns with few unique values might be categorical/targets?
    # For now, just list all columns as potential features.
    columns = list(df.columns)
    
    # Attempt to identify numeric vs categorical
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    
    metadata = {
        "columns": columns,
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "row_count": len(df),
        "suspected_target": _guess_target_column(df),
    }
    
    dataset.metadata_info = metadata
    
    # 2. Chunking strategy
    # For tabular data, a "chunk" could be a JSON representation of say 10 rows, 
    # or a textual description of each row for LLM consumption.
    # Let's do row-level textualization for "Chat Context".
    
    chunks = []
    chunk_size = 5 # Group 5 rows per chunk to avoid too many DB entries
    
    # Clean NaN values before conversion to JSON-friendly dictionary
    df = df.where(pd.notnull(df), None)
    records = df.to_dict('records')
    
    current_chunk_content = []
    chunk_idx = 0
    
    for i, row in enumerate(records):
        # Convert row to text representation: "Age is 50, BP is 120..."
        # or just JSON string
        row_str = json.dumps(row) 
        current_chunk_content.append(row_str)
        
        if len(current_chunk_content) >= chunk_size:
            # Flusing chunk
            chunk_text = "\n".join(current_chunk_content)
            new_chunk = DatasetChunk(
                dataset_id=dataset.id,
                chunk_index=chunk_idx,
                content=chunk_text
            )
            chunks.append(new_chunk)
            
            chunk_idx += 1
            current_chunk_content = []
            
    # Remaining
    if current_chunk_content:
        chunk_text = "\n".join(current_chunk_content)
        new_chunk = DatasetChunk(
            dataset_id=dataset.id,
            chunk_index=chunk_idx,
            content=chunk_text
        )
        chunks.append(new_chunk)
        
    db.add_all(chunks)
    
    dataset.is_processed = True
    db.commit()
    db.refresh(dataset)
    
    return dataset
