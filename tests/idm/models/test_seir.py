import os
import pytest
import textwrap
import yaml

import pandas as pd
from fabric.idm.models import SEIRModel


def test_seir_model():

    model = SEIRModel()

    checkpoints = {"t": ["2020-04-01", "2020-06-01"], "beta": [0.12, 0.1]}
    seir_result, init_params = model.run(start_date="2020-02-01", t=200, checkpoints=checkpoints, verbose=True)

    expected_columns = ["date_id", "S", "E", "I", "D_E", "D_I", "R", "F"]

    assert seir_result.columns.tolist() == expected_columns
    assert len(seir_result.index.tolist()) == 201
    assert init_params is not None
