import sys
from pathlib import Path

# Ensure project root (contains data_types.py) is importable
PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from data_types import PairwiseExample
from evaluation.judge import OnlineJudge, JudgeConfig


def build_config() -> JudgeConfig:
    """Create JudgeConfig for real API call."""
    return JudgeConfig(
        api_key="sk-nzgar7nOKKa6LNwB2cC5cERSirwQxKoXpz4xIX6mKsrzoDEE",
        base_url="https://www.dmxapi.com/v1",
        model="gpt-4o-mini",
        temperature=0.1,
        max_tokens=32,
        timeout=30,
    )


def build_example() -> PairwiseExample:
    """Create concrete pairwise example."""
    return PairwiseExample(
        question_id="q1",
        instruction="Explain why the sky appears blue during the day.",
        response_a=(
            "The sky looks blue because molecules in the atmosphere scatter "
            "shorter blue wavelengths of sunlight more than red wavelengths."
        ),
        response_b=(
            "It is blue because oceans reflect their color upward and the "
            "air absorbs most other light."
        ),
        model_a="science_bot_v1",
        model_b="casual_bot_v2",
    )


def print_flow(prompt: str, raw_output: str, preference: int, confidence: float) -> None:
    print("0) GPT Input Prompt:\n", prompt)
    print("1) GPT Raw Output:\n", raw_output)
    print("2) Parsed Result:\n", {"preference": preference, "confidence": confidence})


def main() -> None:
    config = build_config()
    example = build_example()
    judge = OnlineJudge(config)

    prompt = judge.get_judge_prompt(example)
    result = judge.judge_pairwise(example)
    print_flow(prompt, result.raw_response, result.preference, result.confidence)


if __name__ == "__main__":
    main()

