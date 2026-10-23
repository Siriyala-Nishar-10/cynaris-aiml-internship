"""Compatibility shim for ragas 0.4.x with langchain-community 0.4.x.

ragas imports Vertex AI classes from langchain_community at start-up, but
langchain-community 0.4 no longer ships them (ModuleNotFoundError:
langchain_community.chat_models.vertexai). This project only uses Ollama, so
we register inert placeholder classes for those names. Call apply() BEFORE
importing ragas.
"""
import importlib
import sys
import types


def _placeholder(name):
    return type(name, (), {})


def _tolerate_missing(module_name):
    """Make `from <module_name> import <anything missing>` return a placeholder class."""
    module = importlib.import_module(module_name)
    original = getattr(module, "__getattr__", None)

    def patched(name):
        if original is not None:
            try:
                return original(name)
            except (AttributeError, ImportError):
                pass
        if name.startswith("__"):
            raise AttributeError(name)
        return _placeholder(name)

    module.__getattr__ = patched


def apply():
    try:
        importlib.import_module("langchain_community.chat_models.vertexai")
    except ImportError:
        shim = types.ModuleType("langchain_community.chat_models.vertexai")
        shim.__getattr__ = _placeholder
        sys.modules["langchain_community.chat_models.vertexai"] = shim
    for name in ("langchain_community.llms", "langchain_community.chat_models",
                 "langchain_community.embeddings"):
        try:
            _tolerate_missing(name)
        except ImportError:
            pass