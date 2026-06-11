import re
import sys

import torch


MIN_TORCH_VERSION = (2, 6)


def _version_tuple(version: str) -> tuple[int, ...]:
    match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", version)
    if not match:
        return ()
    return tuple(int(part) for part in match.groups(default="0"))


def require_safe_torch_load() -> None:
    """Require a torch version accepted by current Transformers pickle loading."""
    current = _version_tuple(torch.__version__)
    if current < MIN_TORCH_VERSION:
        raise RuntimeError(
            "torch >= 2.6 is required by current Transformers when loading "
            "pickle-based checkpoints such as facebook/nllb-200-distilled-600M. "
            f"Found torch {torch.__version__}. Upgrade the cluster venv before training."
        )


def main() -> None:
    print(f"torch: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda device: {torch.cuda.get_device_name(0)}")
        print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")
    require_safe_torch_load()
    print("torch pickle-load safety check: OK")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
