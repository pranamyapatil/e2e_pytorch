from __future__ import annotations


class Registry:
    """
    Maps (namespace, name) → class for config-driven instantiation.

    Usage:
        @Registry.register("model", "transformer")
        class TransformerModel(BaseModel): ...

        cls = Registry.get("model", "transformer")
        model = cls(config)
    """

    _registries: dict[str, dict[str, type]] = {}

    @classmethod
    def register(cls, namespace: str, name: str):
        def decorator(klass: type) -> type:
            cls._registries.setdefault(namespace, {})[name] = klass
            klass._registry_name = name
            klass._registry_namespace = namespace
            return klass
        return decorator

    @classmethod
    def get(cls, namespace: str, name: str) -> type:
        available = list(cls._registries.get(namespace, {}).keys())
        if name not in cls._registries.get(namespace, {}):
            raise KeyError(
                f"'{name}' not registered under '{namespace}'. "
                f"Available: {available}"
            )
        return cls._registries[namespace][name]

    @classmethod
    def build(cls, namespace: str, name: str, config) -> object:
        return cls.get(namespace, name)(config)

    @classmethod
    def list(cls, namespace: str) -> list[str]:
        return list(cls._registries.get(namespace, {}).keys())
