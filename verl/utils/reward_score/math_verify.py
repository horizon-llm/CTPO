# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

_MATH_VERIFY_IMPORT_ERROR = None
try:
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
except ImportError as e:
    _MATH_VERIFY_IMPORT_ERROR = e


def _normalize_ground_truth(ground_truth: str) -> str:
    """Normalize common dataset artifacts before math-verify parsing.

    Some datasets contain integers with leading zeros (e.g. "080"), which may
    fail parser-side expression handling. For purely integer strings, normalize
    them to canonical decimal form while preserving sign.
    """
    text = str(ground_truth).strip()
    if not text:
        return text

    sign = ""
    body = text
    if body[0] in "+-":
        sign = body[0]
        body = body[1:]

    if body.isdigit():
        # int("000") -> 0, int("080") -> 80
        return f"{sign}{int(body)}"
    return text


def _build_gold_candidates(ground_truth: str) -> list[str]:
    """Build robust gold-target candidates for math-verify.

    We avoid forcing one single representation (e.g., always boxed), because
    different datasets may provide gold answers in slightly different formats.
    """
    raw = str(ground_truth).strip()
    normalized = _normalize_ground_truth(raw)

    candidates: list[str] = []
    for c in (raw, normalized):
        c = c.strip()
        if c and c not in candidates:
            candidates.append(c)

    # If wrapped by inline math delimiters, also try stripped content.
    if normalized.startswith("$") and normalized.endswith("$") and len(normalized) >= 2:
        inner = normalized[1:-1].strip()
        if inner and inner not in candidates:
            candidates.append(inner)

    # Backward-compatible candidate: boxed form when not already boxed.
    if normalized and "\\boxed{" not in normalized:
        boxed = "\\boxed{" + normalized + "}"
        if boxed not in candidates:
            candidates.append(boxed)

    return candidates


def compute_score(model_output: str, ground_truth: str, timeout_score: float = 0) -> bool:
    if _MATH_VERIFY_IMPORT_ERROR is not None:
        raise RuntimeError(
            "math-verify is required but not available. "
            "Please install it with `pip install math-verify`."
        ) from _MATH_VERIFY_IMPORT_ERROR

    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    best_score = 0.0
    errors: list[str] = []
    candidates = _build_gold_candidates(ground_truth)

    for gold in candidates:
        try:
            ret_score, _ = verify_func([gold], [model_output])
            best_score = max(best_score, float(ret_score))
            if best_score >= 1.0:
                return best_score
        except TimeoutException:
            best_score = max(best_score, float(timeout_score))
        except Exception as e:
            errors.append(f"{type(e).__name__} on candidate={gold!r}: {e}")

    if errors and best_score <= 0:
        raise RuntimeError(
            "math-verify scoring failed for all candidates. "
            f"ground_truth={ground_truth!r}, candidates={candidates!r}, "
            f"errors={errors[:3]!r}"
        )

    return best_score
