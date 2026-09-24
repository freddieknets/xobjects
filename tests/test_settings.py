# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

import xobjects as xo
import numpy as np
import pytest

from xobjects.test_helpers import for_all_test_contexts


def test_cuda_compute_capability():
    assert xo.ContextCupy().cuda_compute_capability is None
    with xo.settings.override(cuda_compute_capability=90):
        assert xo.ContextCupy().cuda_compute_capability == 90
        assert (
            xo.ContextCupy(cuda_compute_capability=80).cuda_compute_capability
            == 80
        )
