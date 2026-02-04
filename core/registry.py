"""Lightweight factory/registry helpers for datasets, models, and voting strategies."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, TypeVar

T = TypeVar("T")


class Registry:
    """Simple string-to-callable registry used by factory helpers."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._items: Dict[str, Callable[..., Any]] = {}

    def register(self, key: Optional[str] = None) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Decorator to register a callable/class under a unique key."""

        def decorator(target: Callable[..., T]) -> Callable[..., T]:
            registry_key = (key or target.__name__).lower()
            if registry_key in self._items:
                raise ValueError(f"{self._name} '{registry_key}' already registered")
            self._items[registry_key] = target
            return target

        return decorator

    def get(self, key: str) -> Callable[..., Any]:
        registry_key = key.lower()
        if registry_key not in self._items:
            available = ", ".join(sorted(self._items)) or "<empty>"
            raise KeyError(f"Unknown {self._name} '{key}'. Available: {available}")
        return self._items[registry_key]

    def create(self, key: str, **kwargs: Any) -> Any:
        target = self.get(key)
        return target(**kwargs)

    def available(self) -> list[str]:
        return sorted(self._items.keys())


model_registry = Registry("model")
dataset_registry = Registry("dataset")
voting_registry = Registry("voting strategy")


def register_model(key: Optional[str] = None) -> Callable[[Callable[..., T]], Callable[..., T]]:
    return model_registry.register(key)


def register_dataset(key: Optional[str] = None) -> Callable[[Callable[..., T]], Callable[..., T]]:
    return dataset_registry.register(key)


def register_voting(key: Optional[str] = None) -> Callable[[Callable[..., T]], Callable[..., T]]:
    return voting_registry.register(key)


def create_model(key: str, **kwargs: Any) -> Any:
    return model_registry.create(key, **kwargs)


def create_dataset(key: str, **kwargs: Any) -> Any:
    return dataset_registry.create(key, **kwargs)


def create_voting_strategy(key: str, **kwargs: Any) -> Any:
    return voting_registry.create(key, **kwargs)

