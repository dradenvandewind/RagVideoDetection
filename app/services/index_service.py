
import asyncio

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore

from ..core.config import Settings


def get_chroma_client(settings: Settings) -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=settings.chroma_path)


def get_chroma_collection(settings: Settings):
    client = get_chroma_client(settings)
    return client.get_or_create_collection(settings.chroma_collection)


def _build_index_sync(settings: Settings) -> VectorStoreIndex:
    chroma_collection = get_chroma_collection(settings)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
    )


async def build_index(settings: Settings) -> VectorStoreIndex:
    """Version async utilisée au démarrage (lifespan) et lors d'un reset."""
    return await asyncio.to_thread(_build_index_sync, settings)


async def reset_index(settings: Settings) -> VectorStoreIndex:
    """Supprime puis recrée la collection Chroma, renvoie un index neuf."""
    client = get_chroma_client(settings)
    await asyncio.to_thread(client.delete_collection, settings.chroma_collection)
    await asyncio.to_thread(client.get_or_create_collection, settings.chroma_collection)
    return await build_index(settings)


async def count_chunks(settings: Settings) -> int:
    collection = get_chroma_collection(settings)
    return await asyncio.to_thread(collection.count)