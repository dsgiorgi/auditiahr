# utils/blob_storage.py
import os
from pathlib import Path
from typing import Optional
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceExistsError
from .kv_client import get_secret

CONNECTION_SECRET_NAME = os.getenv("STORAGE_CONN_SECRET", "StorageConnection")
CONTAINER_NAME = os.getenv("BLOB_CONTAINER", "auditiahrssession")

_conn_str = get_secret(CONNECTION_SECRET_NAME)
_blob_service = BlobServiceClient.from_connection_string(_conn_str)
_container = _blob_service.get_container_client(CONTAINER_NAME)

def ensure_container() -> None:
    try:
        _container.create_container()
    except ResourceExistsError:
        pass

def upload_bytes(blob_name: str, data: bytes) -> str:
    ensure_container()
    _container.upload_blob(name=blob_name, data=data, overwrite=True)
    return blob_name

def upload_file(file_path: str, blob_name: Optional[str] = None) -> str:
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    name = blob_name or p.name
    with p.open("rb") as f:
        upload_bytes(name, f.read())
    return name

def download_bytes(blob_name: str) -> bytes:
    return _container.download_blob(blob_name).readall()

def download_to_file(blob_name: str, dest_path: str) -> str:
    p = Path(dest_path)
    data = download_bytes(blob_name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return str(p)

def delete_blob(blob_name: str) -> None:
    try:
        _container.delete_blob(blob_name)
    except Exception:
        pass
