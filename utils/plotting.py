"""
Executes LLM-generated plotting code against a query result DataFrame in a restricted
namespace, saves the figure to disk, and returns both the file path and a base64
data URI (handy for API/UI consumption without a filesystem round trip).
"""
from __future__ import annotations

import base64
import io
import logging
import uuid
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless backend, no display required
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from config.settings import settings

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

# Only these names are exposed to generated plotting code.
_SAFE_GLOBALS: dict[str, Any] = {
    "pd": pd,
    "plt": plt,
    "sns": sns,
}

_FORBIDDEN_TOKENS = (
    "import os", "import sys", "subprocess", "open(", "exec(", "eval(",
    "__import__", "os.", "sys.", "shutil", "socket", "requests",
)


class UnsafePlotCodeError(Exception):
    pass


def _validate_code(code: str) -> None:
    lowered = code.lower()
    for token in _FORBIDDEN_TOKENS:
        if token in lowered:
            raise UnsafePlotCodeError(f"Generated plot code contains disallowed token: '{token}'")


def render_plot(df: pd.DataFrame, plot_code: str, title_hint: str = "chart") -> dict[str, str]:
    """
    `plot_code` must be a snippet that uses the pre-bound variable `df` and the
    exposed `plt`/`sns`/`pd` names to build a figure on the current matplotlib axes.
    It should NOT call plt.show() or plt.savefig() itself — this function handles saving.
    """
    _validate_code(plot_code)

    plt.close("all")
    fig = plt.figure(figsize=(8, 5))

    local_scope = {"df": df}
    try:
        exec(compile(plot_code, "<generated_plot>", "exec"), dict(_SAFE_GLOBALS), local_scope)  # noqa: S102
    except Exception as e:  # noqa: BLE001
        plt.close(fig)
        raise RuntimeError(f"Failed to execute generated plot code: {e}") from e

    plt.tight_layout()

    output_dir = Path(settings.plots_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{title_hint.replace(' ', '_')[:40]}_{uuid.uuid4().hex[:8]}.png"
    filepath = output_dir / filename
    fig.savefig(filepath, dpi=150, bbox_inches="tight")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    logger.info("Saved plot to %s", filepath)
    return {
        "file_path": str(filepath),
        "base64_png": b64,
        "data_uri": f"data:image/png;base64,{b64}",
    }
