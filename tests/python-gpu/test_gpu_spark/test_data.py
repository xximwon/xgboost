import pytest

from xgboost import testing

skip = testing.skip_spark()
if skip["condition"]:
    pytest.skip(msg=skip["reason"], allow_module_level=True)

testing.add_path("tests/python")

import testing as tm
from test_spark.test_data import run_dmatrix_ctor


@pytest.mark.skipif(**tm.no_cudf())
def test_qdm_ctor() -> None:
    run_dmatrix_ctor(True)
