# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=protected-access, c-extension-no-member
"""Hooks to PyTorch."""
import torch

import _RAZORC


def _to(self, *args, **kwargs):
    ret = super(torch.nn.parameter.Parameter, self).to(*args, **kwargs)
    if str(ret.device.type) == "lazy":
        return _RAZORC._raf_mark_parameter(ret)
    return ret


torch.nn.parameter.Parameter.to = _to
