"""
Model-agnostic LLM + embeddings factory.

Every agent/graph node calls `get_chat_llm()` / `get_embeddings()` instead of
instantiating a specific LangChain client. Swapping providers is purely a `.env` change:

    LLM_PROVIDER=ollama      -> ChatOllama            (local dev)
    LLM_PROVIDER=groq        -> ChatGroq              (API dev/test)
    LLM_PROVIDER=qwen_vllm   -> ChatOpenAI (OpenAI-compatible endpoint) (production, multi-GPU)
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from config.settings import settings


@lru_cache(maxsize=1)
def get_chat_llm(temperature: float | None = None) -> Any:
    """Returns a LangChain chat-model instance for the configured provider."""
    temp = settings.llm_temperature if temperature is None else temperature
    provider = settings.llm_provider.lower()

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=temp,
            num_predict=settings.llm_max_tokens,
        )

    if provider == "groq":
        from langchain_groq import ChatGroq

        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY is not set in .env")
        return ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            temperature=temp,
            max_tokens=settings.llm_max_tokens,
        )

    if provider == "qwen_vllm":
        # Any OpenAI-compatible server (vLLM, TGI, SGLang) works through ChatOpenAI.
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            base_url=settings.qwen_base_url,
            api_key=settings.qwen_api_key,
            model=settings.qwen_model,
            temperature=temp,
            max_tokens=settings.llm_max_tokens,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER '{provider}'. Expected one of: ollama, groq, qwen_vllm"
    )


@lru_cache(maxsize=1)
def get_embeddings() -> Any:
    """Returns a LangChain embeddings instance for the configured embedding provider."""
    provider = settings.embedding_provider.lower()

    if provider == "sentence_transformers":
        from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=settings.sentence_transformers_model)

    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(base_url=settings.ollama_base_url, model=settings.ollama_embed_model)

    if provider == "openai_compatible":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            base_url=settings.embed_base_url,
            api_key="not-needed",
            model=settings.embed_model,
        )

    raise ValueError(
        f"Unknown EMBEDDING_PROVIDER '{provider}'. "
        "Expected one of: sentence_transformers, ollama, openai_compatible"
    )
