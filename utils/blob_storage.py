# utils/blob_storage.py
import os
from pathlib import Path
from typing import Optional

from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceExistsError

from .kv_client import get_secret

CONNECTION_SECRET_NAME = os.getenv("STORAGE_CONN_SECRET", "StorageConnection")
CONTAINER_NAME = os.getenv("BLOB_CONTAINER", "auditiahrsession")


def _get_blob_service() -> BlobServiceClient:
    conn_str = get_secret(CONNECTION_SECRET_NAME)
    return BlobServiceClient.from_connection_string(conn_str)


_blob_service = _get_blob_service()
_container = _blob_service.get_container_client(CONTAINER_NAME)


def ensure_container() -> None:
    try:
        _container.create_container()
    except ResourceExistsError:
        pass


def upload_bytes(blob_name: str, data: bytes, content_type: str = "text/csv") -> str:
    ensure_container()
    blob = _container.get_blob_client(blob_name)
    blob.upload_blob(data, overwrite=True, content_type=content_type)
    return blob.url


def upload_file(blob_name: str, file_path: str, content_type: Optional[str] = None) -> str:
    p = Path(file_path)
    data = p.read_bytes()
    return upload_bytes(blob_name, data, content_type=content_type)


def download_bytes(blob_name: str) -> bytes:
    ensure_container()
    blob = _container.get_blob_client(blob_name)
    return blob.download_blob().readall()


def download_to_file(blob_name: str, dest_path: str) -> str:
    p = Path(dest_path)
    data = download_bytes(blob_name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return str(p)


def delete_blob(blob_name: str) -> None:
    try:
        blob = _container.get_blob_client(blob_name)
        blob.delete_blob()
    except Exception:
        # No rompemos si el blob ya no existe
        pass
