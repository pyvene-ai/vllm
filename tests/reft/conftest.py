# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os as _os

import torch as _torch

if not _torch.cuda.is_available():
    # No GPUs visible (e.g. a CUDA build on a GPU-less node): force the
    # CPU platform so the ReFT unit tests can run.  Must happen before
    # vllm.platforms.current_platform is first resolved.
    _os.environ.setdefault("VLLM_FORCE_PLATFORM", "cpu")
