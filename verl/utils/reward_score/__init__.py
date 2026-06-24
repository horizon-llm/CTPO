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

from verl.utils.import_utils import deprecated


def _extract_numeric_score(res) -> float:
    if isinstance(res, dict):
        if "score" in res:
            return float(res["score"])
        return 0.0
    if isinstance(res, int | float | bool):
        return float(res)
    return float(res[0])


def _to_binary_hit(res) -> float:
    return 1.0 if _extract_numeric_score(res) > 0 else 0.0


def default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
    **kwargs,
):
    # Ensemble policy: any scorer hits => correct.
    # We keep this data_source-agnostic on purpose for your current workflow.
    from . import math_dapo, math_reward, math_verify

    hits: list[float] = []

    try:
        hits.append(_to_binary_hit(math_reward.compute_score(solution_str, ground_truth)))
    except Exception:
        pass

    try:
        hits.append(_to_binary_hit(math_dapo.compute_score(solution_str, ground_truth)))
    except Exception:
        pass

    try:
        hits.append(_to_binary_hit(math_verify.compute_score(solution_str, ground_truth)))
    except Exception:
        pass

    if not hits:
        return 0.0
    return max(hits)


@deprecated("verl.utils.reward_score.default_compute_score")
def _default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
):
    return default_compute_score(
        data_source, solution_str, ground_truth, extra_info, sandbox_fusion_url, concurrent_semaphore, memory_limit_mb
    )


__all__ = ["default_compute_score"]
