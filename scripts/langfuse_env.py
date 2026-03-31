import os


def get_first_env(*names: str) -> str | None:
    for name in names:
        if not name:
            continue
        value = os.getenv(name)
        if value:
            return value
    return None


def get_langfuse_host() -> str | None:
    return get_first_env("LANGFUSE_HOST", "LANGFUSE_BASE_URL")


def require_env(name: str, *, context: str | None = None) -> str:
    value = os.getenv(name)
    if value:
        return value
    detail = f" for {context}" if context else ""
    raise ValueError(f"Missing required environment variable {name}{detail}.")
