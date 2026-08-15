# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os as _os

import torch as _torch

if not _torch.cuda.is_available():
    # No GPUs visible: force the CPU platform so the adaptation unit
    # tests can run.  Must happen before current_platform resolves.
    _os.environ.setdefault("VLLM_FORCE_PLATFORM", "cpu")
