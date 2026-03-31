import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from langfuse import Langfuse
from langfuse.experiment import Evaluation
from langfuse.openai import openai as lf_openai
from langfuse_env import get_langfuse_host, require_env

load_dotenv()

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
JUDGE_SYSTEM_PROMPT_PATH = PROMPTS_DIR / "langfuse_judge_system_prompt.txt"
DEFAULT_API_PROFILE_PATH = Path(__file__).resolve().parents[1] / "configs" / "model_api_profiles.json"


def load_prompt_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def load_api_profiles(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    profiles = data.get("profiles", data)
    return profiles if isinstance(profiles, dict) else {}


def make_langfuse_client() -> Langfuse:
    public_key = require_env("LANGFUSE_PUBLIC_KEY", context="Langfuse client")
    secret_key = require_env("LANGFUSE_SECRET_KEY", context="Langfuse client")
    host = get_langfuse_host()
    return Langfuse(public_key=public_key, secret_key=secret_key, host=host)


def make_model_client(
    *,
    api_key_env: str = "MODEL_API_KEY",
    default_base_url: str = "https://api.openai.com/v1",
    explicit_base_url: str | None = None,
    base_url_env: str | None = "MODEL_BASE_URL",
) -> Any:
    api_key = os.getenv(api_key_env)
    base_url = explicit_base_url or (os.getenv(base_url_env, default_base_url) if base_url_env else default_base_url)
    if not api_key:
        raise ValueError(
            f"Missing required environment variable {api_key_env} for model client "
            f"(base_url={base_url})."
        )
    return lf_openai.OpenAI(api_key=api_key, base_url=base_url)


def render_context(turns: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for turn in turns:
        timestamp = turn.get("timestamp") or ""
        speaker = turn.get("speaker") or "Unknown"
        text = turn.get("text") or ""
        voice_style = turn.get("voice_style")
        line = f"[{timestamp}] {speaker}: {text}"
        if voice_style not in {None, ""}:
            line += f"\nvoice_style: {voice_style}"
        lines.append(line)
    return "\n\n".join(lines)


def render_options(options: List[str]) -> str:
    if not options:
        return ""
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    rendered = [f"{letters[index]}. {option}" for index, option in enumerate(options)]
    return "\n".join(rendered)


def normalize_answer(value: str) -> str:
    text = value.strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    if text.startswith("{"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and isinstance(parsed.get("answer"), str):
                text = parsed["answer"]
        except Exception:
            pass
    text = re.sub(r"^[A-Z][\)\.\:\-]\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def safe_print(value: str) -> None:
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        print(value)
    except UnicodeEncodeError:
        print(value.encode(encoding, errors="replace").decode(encoding, errors="replace"))


def get_input_value(item: Any) -> Dict[str, Any]:
    return item["input"] if isinstance(item, dict) else item.input


def get_expected_output_value(item: Any) -> Dict[str, Any]:
    return item.get("expected_output") if isinstance(item, dict) else item.expected_output


def get_metadata_value(item: Any) -> Dict[str, Any]:
    return item.get("metadata") if isinstance(item, dict) else item.metadata


def parse_judge_response(content: str) -> Dict[str, Any]:
    text = content.strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return {
        "core_answer_correct": 0.0,
        "key_point_coverage": 0.0,
        "rationale_alignment": 0.0,
        "unsupported_claims": 0.0,
        "insufficiency_handling": 0.0,
        "final_score": 0.0,
        "verdict": "parse_error",
        "comment": f"Could not parse judge output: {text[:200]}",
    }


def run_client_preflight(*, client: Any, model_name: str, label: str) -> None:
    try:
        response = client.chat.completions.create(
            model=model_name,
            temperature=0.0,
            messages=[
                {"role": "system", "content": "Reply with OK."},
                {"role": "user", "content": "OK"},
            ],
            max_tokens=8,
        )
        _ = response.choices[0].message.content
    except Exception as exc:
        raise RuntimeError(f"{label} preflight failed for model {model_name}: {exc}") from exc


def resolve_client_settings(
    *,
    profiles: Dict[str, Dict[str, Any]],
    profile_name: str | None,
    cli_model: str | None,
    cli_api_key_env: str | None,
    cli_base_url_env: str | None,
    cli_base_url: str | None,
    default_model: str,
    default_api_key_env: str,
    default_base_url_env: str,
    default_base_url: str,
) -> Dict[str, str]:
    profile = profiles.get(profile_name, {}) if profile_name else {}
    return {
        "model": cli_model or profile.get("model") or default_model,
        "api_key_env": cli_api_key_env or profile.get("api_key_env") or default_api_key_env,
        "base_url_env": cli_base_url_env or profile.get("base_url_env") or default_base_url_env,
        "base_url": cli_base_url or profile.get("base_url") or default_base_url,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Langfuse experiment on uploaded benchmark items.")
    parser.add_argument("--dataset-name", required=True, help="Langfuse dataset name")
    parser.add_argument("--prompt-name", required=True, help="Langfuse prompt name")
    parser.add_argument("--prompt-label", default="production", help="Langfuse prompt label")
    parser.add_argument("--context-policy", choices=["session_local", "full_history"], required=True)
    parser.add_argument("--api-profile-path", default=str(DEFAULT_API_PROFILE_PATH), help="JSON config path for model provider profiles")
    parser.add_argument("--task-profile", default=None, help="Task model profile name from the API profile config")
    parser.add_argument("--judge-profile", default=None, help="Judge model profile name from the API profile config")
    parser.add_argument("--model", default=None, help="Optional explicit task model override")
    parser.add_argument("--judge-model", default=None, help="Optional explicit judge model override")
    parser.add_argument("--task-api-key-env", default=None, help="Optional env var name for the task model API key")
    parser.add_argument("--task-base-url", default=None, help="Optional explicit base URL for the task model client")
    parser.add_argument("--task-base-url-env", default=None, help="Optional env var name for the task model base URL")
    parser.add_argument("--judge-api-key-env", default=None, help="Optional env var name for the judge model API key")
    parser.add_argument("--judge-base-url", default=None, help="Optional explicit base URL for the judge model client")
    parser.add_argument("--judge-base-url-env", default=None, help="Optional env var name for the judge model base URL")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--skip-preflight", action="store_true", help="Skip task/judge model connectivity preflight")
    parser.add_argument("--max-items", type=int, default=None, help="Optional limit on the number of dataset items to evaluate")
    parser.add_argument("--experiment-name", default="emotion-memory-eval")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--report-path", default=None, help="Optional path to save a local text report")
    args = parser.parse_args()

    langfuse = make_langfuse_client()
    profiles = load_api_profiles(Path(args.api_profile_path))
    task_settings = resolve_client_settings(
        profiles=profiles,
        profile_name=args.task_profile,
        cli_model=args.model,
        cli_api_key_env=args.task_api_key_env,
        cli_base_url_env=args.task_base_url_env,
        cli_base_url=args.task_base_url,
        default_model=os.getenv("MODEL_NAME", "@gemini/gemini-2.5-flash"),
        default_api_key_env="MODEL_API_KEY",
        default_base_url_env="MODEL_BASE_URL",
        default_base_url="https://api.openai.com/v1",
    )
    judge_settings = resolve_client_settings(
        profiles=profiles,
        profile_name=args.judge_profile,
        cli_model=args.judge_model,
        cli_api_key_env=args.judge_api_key_env,
        cli_base_url_env=args.judge_base_url_env,
        cli_base_url=args.judge_base_url,
        default_model=task_settings["model"],
        default_api_key_env=task_settings["api_key_env"],
        default_base_url_env=task_settings["base_url_env"],
        default_base_url=task_settings["base_url"],
    )
    task_model_client = make_model_client(
        api_key_env=task_settings["api_key_env"],
        explicit_base_url=task_settings["base_url"],
        base_url_env=task_settings["base_url_env"],
    )
    judge_model_client = make_model_client(
        api_key_env=judge_settings["api_key_env"],
        explicit_base_url=judge_settings["base_url"],
        base_url_env=judge_settings["base_url_env"],
    )
    task_model_name = task_settings["model"]
    judge_model_name = judge_settings["model"]
    judge_system_prompt = load_prompt_text(JUDGE_SYSTEM_PROMPT_PATH)
    dataset = langfuse.get_dataset(args.dataset_name)
    prompt = langfuse.get_prompt(args.prompt_name, label=args.prompt_label, type="chat")

    if not args.skip_preflight:
        run_client_preflight(client=task_model_client, model_name=task_model_name, label="Task model")
        run_client_preflight(client=judge_model_client, model_name=judge_model_name, label="Judge model")

    def task(*, item: Any, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        input_data = get_input_value(item)
        benchmark_views = input_data["benchmark_views"]
        modality_views = benchmark_views.get("modality_conditioned_views")

        if modality_views:
            condition_payload = next(iter(modality_views.values()))
            context_turns = condition_payload[f"{args.context_policy}_context"]
        else:
            view_key = "session_local_context" if args.context_policy == "session_local" else "full_history_context"
            context_turns = benchmark_views[view_key]

        question_text = input_data["question_text"]
        options_block = render_options(input_data.get("options", []))
        context_transcript = render_context(context_turns)

        messages = prompt.compile(
            question_text=question_text,
            context_transcript=context_transcript,
            options_block=options_block,
            context_policy=args.context_policy,
        )
        response = task_model_client.chat.completions.create(
            model=task_model_name,
            temperature=args.temperature,
            messages=messages,
        )
        content = response.choices[0].message.content or ""
        return {
            "answer": content.strip(),
            "context_policy": args.context_policy,
        }

    def exact_match_evaluator(
        *,
        input: Dict[str, Any],
        output: Dict[str, Any],
        expected_output: Dict[str, Any],
        metadata: Dict[str, Any] | None,
        **kwargs: Dict[str, Any],
    ) -> Evaluation | List[Evaluation]:
        if not input.get("options"):
            return []
        predicted = normalize_answer(str(output.get("answer", "")))
        candidates = [expected_output.get("gold_answer", ""), *expected_output.get("acceptable_answers", [])]
        normalized_candidates = [normalize_answer(str(candidate)) for candidate in candidates if str(candidate).strip()]
        is_match = predicted in normalized_candidates
        return Evaluation(
            name="exact_match",
            value=1.0 if is_match else 0.0,
            comment=f"pred={predicted!r}, golds={normalized_candidates!r}",
        )

    def llm_judge_evaluator(
        *,
        input: Dict[str, Any],
        output: Dict[str, Any],
        expected_output: Dict[str, Any],
        metadata: Dict[str, Any] | None,
        **kwargs: Dict[str, Any],
    ) -> Evaluation | List[Evaluation]:
        if input.get("options"):
            return []

        judge_payload = {
            "question_text": input.get("question_text"),
            "model_answer": output.get("answer"),
            "gold_answer": expected_output.get("gold_answer"),
            "acceptable_answers": expected_output.get("acceptable_answers", []),
            "key_explanation_points": expected_output.get("key_explanation_points", []),
            "gold_rationale": expected_output.get("gold_rationale"),
            "metadata": metadata or {},
        }
        response = judge_model_client.chat.completions.create(
            model=judge_model_name,
            temperature=0.0,
            messages=[
                {"role": "system", "content": judge_system_prompt},
                {"role": "user", "content": json.dumps(judge_payload, ensure_ascii=False, indent=2)},
            ],
        )
        parsed = parse_judge_response(response.choices[0].message.content or "")
        verdict = str(parsed.get("verdict", "judge"))
        comment = str(parsed.get("comment", ""))

        def to_float(key: str) -> float:
            try:
                return float(parsed.get(key, 0.0))
            except Exception:
                return 0.0

        return [
            Evaluation(name="judge_core_answer_correct", value=to_float("core_answer_correct"), comment=comment),
            Evaluation(name="judge_key_point_coverage", value=to_float("key_point_coverage"), comment=comment),
            Evaluation(name="judge_rationale_alignment", value=to_float("rationale_alignment"), comment=comment),
            Evaluation(name="judge_unsupported_claims", value=to_float("unsupported_claims"), comment=comment),
            Evaluation(name="judge_insufficiency_handling", value=to_float("insufficiency_handling"), comment=comment),
            Evaluation(name="llm_judge_final", value=to_float("final_score"), comment=f"{verdict}: {comment}"),
        ]

    if args.max_items is not None:
        original_items = list(getattr(dataset, "items", []) or [])
        dataset.items = original_items[: args.max_items]

    result = dataset.run_experiment(
        name=args.experiment_name,
        run_name=args.run_name,
        task=task,
        evaluators=[exact_match_evaluator, llm_judge_evaluator],
    )

    safe_print(result.format(include_item_results=False))
    if result.dataset_run_url:
        safe_print(f"Langfuse run URL: {result.dataset_run_url}")

    if args.report_path:
        Path(args.report_path).write_text(
            result.format(include_item_results=True),
            encoding="utf-8",
        )
        safe_print(f"Saved local report to: {args.report_path}")


if __name__ == "__main__":
    main()
