import math
import os
import pytest
import textwrap
import yaml
from shutil import copyfile
import pdb

import pandas as pd
from fabric.idm.models import RTModel
from fabric.idm.utility import load_data

@pytest.fixture
def input_data(tmpdir):

    src = "tests/resources/prepare_data_two.parquet"
    dest = os.path.join(tmpdir, "src_confirmed_rt.parquet")

    copyfile(src, dest)
    return dest

@pytest.fixture
def init_model():
    return RTModel(start_date="2020-04-01")

def test_run(input_data, init_model):
    cases = pd.read_parquet(input_data)
    data = init_model.run(cases)
    expected_columns = ["date_id", "geo_id", "r_t_most_likely", "r_t_ci_5", "r_t_ci_95", "case_growth_class"]

    assert data.columns.tolist() == expected_columns
