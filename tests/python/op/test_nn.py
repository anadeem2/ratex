# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn as nn

import ratex
from ratex.testing import verify_step


def test_conv():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(
                in_channels=1, out_channels=6, kernel_size=5, padding=2, bias=False
            )

        def forward(self, x):
            out = self.conv(x)
            return out

    shape = [1, 1, 28, 28]
    x = torch.randn(*shape)
    verify_step(Model(), [x])


def test_linear():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(120, 84)

        def forward(self, x):
            out = self.linear(x)
            return out

    shape = [32, 120]
    x = torch.randn(*shape)
    verify_step(Model(), [x])


def test_sum():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            out = torch.sum(x)
            return out

    shape = [32, 120]
    x = torch.randn(*shape)
    verify_step(Model(), [x], jit_script=False, tol=5e-4)


def test_pad():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            pad = (1, 2, 3, 4, 5, 6)
            out = torch.nn.functional.pad(x, pad, "constant", 2)
            return out

    shape = [32, 120, 20]
    x = torch.randn(*shape)
    verify_step(Model(), [x], jit_script=False)


def test_gelu():
    """GeLU supports approximation since https://github.com/pytorch/pytorch/pull/72826"""

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.gelu = torch.nn.GELU("none")

        def forward(self, x):
            return self.gelu(x)

    shape = [5, 5]
    x = torch.randn(*shape)
    verify_step(Model(), [x], jit_script=False)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("norm_type", [1, 2])
def test_embedding(dtype, norm_type):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(10, 3, norm_type=norm_type, dtype=dtype)

        def forward(self, x_input):
            return self.embedding(x_input)

    x = torch.randint(10, (3, 3))
    verify_step(Model(), [x], jit_script=False)


@pytest.mark.parametrize("shape", [(3, 3, 3)])
def test_matmul(shape):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x_input, y_input):
            return torch.matmul(x_input, y_input)

    # x = torch.randn(3,3)
    # verify_step(Model(), [x, y], jit_script=False)

    # for i in range(1, len(shape) + 1):
    #     x_s = shape[::i]
    #     x = torch.randn(x_s)
    #     for j in range(1, len(shape) + 1):
    #         y_s = shape[::j]
    #         y = torch.randn(y_s)
    #         verify_step(Model(), [x, y], jit_script=False)


def test_softmax():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.softmax = nn.Softmax()

        def forward(self, x_input):
            return self.softmax(x_input)

    x = torch.randn(3, 3)

    verify_step(Model(), [x], jit_script=False)


def test_dropout():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = nn.Dropout(p=0.2)

        def forward(self, x_input):
            return self.dropout(x_input)

    x = torch.randn(20, 16)
   
    verify_step(Model(), [x], jit_script=False)

def test_layer_norm():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_norm = nn.LayerNorm(10)

        def forward(self, x_input):
            return self.layer_norm(x_input)
    
    batch, sentence_length, embedding_dim = 20, 5, 10
    x = torch.randn(batch, sentence_length, embedding_dim)
    

    verify_step(Model(), [x], jit_script=False)

@pytest.mark.parametrize("dtype", [torch.float32])
def test_native_batch_norm(dtype):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, w, b, running_mean, running_var):
            return torch.native_batch_norm(x, w, b, running_mean, running_var, False, 0.9, .001)

    x = torch.randn(3, 3).to(dtype)
    w = torch.randn(3, 3).to(dtype)
    b = torch.randn(3).to(dtype)
    running_mean = torch.randn(3).to(dtype)
    running_var = torch.randn(3).to(dtype)
    verify_step(Model(), [x, w, b, running_mean, running_var], jit_script=False)

if __name__ == "__main__":
    pytest.main([__file__])
