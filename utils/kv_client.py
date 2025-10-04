# utils/kv_client.py
import os
from typing import Optional
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.keyvault.secrets import SecretClient

KV_URL = os.getenv("KV_URL", "https://auditiahr-kv.vault.azure.net/")

def _get_client() -> SecretClient:
    try:
        cred = DefaultAzureCredential()
        client = SecretClient(vault_url=KV_URL, credential=cred)
        _ = next(client.list_properties_of_secrets(), None)
        return client
    except Exception:
        ibc = InteractiveBrowserCredential()
        return SecretClient(vault_url=KV_URL, credential=ibc)

_client_singleton: Optional[SecretClient] = None

def _client() -> SecretClient:
    global _client_singleton
    if _client_singleton is None:
        _client_singleton = _get_client()
    return _client_singleton

def get_secret(name: str) -> str:
    return _client().get_secret(name).value
