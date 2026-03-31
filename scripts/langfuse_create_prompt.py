import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from langfuse import Langfuse
from langfuse_env import get_langfuse_host, require_env

load_dotenv()
PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
SYSTEM_PROMPT_PATH = PROMPTS_DIR / "langfuse_baseline_system_prompt.txt"
USER_PROMPT_PATH = PROMPTS_DIR / "langfuse_baseline_user_prompt.txt"


def load_prompt_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def make_langfuse_client() -> Langfuse:
    public_key = require_env("LANGFUSE_PUBLIC_KEY", context="Langfuse client")
    secret_key = require_env("LANGFUSE_SECRET_KEY", context="Langfuse client")
    host = get_langfuse_host()
    return Langfuse(public_key=public_key, secret_key=secret_key, host=host)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create or update the Langfuse baseline prompt.")
    parser.add_argument("--prompt-name", default="emotion-memory-baseline")
    parser.add_argument("--prompt-label", default="production")
    parser.add_argument(
        "--commit-message",
        default="Baseline prompt for session-local and full-history emotion-memory evaluation",
    )
    args = parser.parse_args()

    langfuse = make_langfuse_client()
    system_prompt = load_prompt_text(SYSTEM_PROMPT_PATH)
    user_prompt = load_prompt_text(USER_PROMPT_PATH)
    langfuse.create_prompt(
        name=args.prompt_name,
        type="chat",
        labels=[args.prompt_label],
        commit_message=args.commit_message,
        prompt=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    print(f"Prompt created/updated: {args.prompt_name} [{args.prompt_label}]")


if __name__ == "__main__":
    main()
