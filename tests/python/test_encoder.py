import numpy as np
import xgboost as xgb


def run_encoder_test(device: str) -> None:
    if device == "cpu":
        import pandas as pd

        Df = pd.DataFrame
        Ser = pd.Series
    else:
        import cudf

        Df = cudf.DataFrame
        Ser = cudf.Series

    df = Df({"c": ["cdef", "abc"]}, dtype="category")
    categories = df.c.cat.categories

    # fixme: test save binary.
    Xy = xgb.DMatrix(df, enable_categorical=True)
    results = Xy.get_categories()
    assert len(results["c"]) == len(categories)
    for i in range(len(results["c"])):
        assert str(results["c"][i]) == str(categories[i]), (
            results["c"][i],
            categories[i],
        )

    y = Ser([1, 2], name="y")
    Xy.set_info(label=y)
    # fixme: test mismached device, DMatrix is used internally.
    booster = xgb.train({"device": device}, Xy, num_boost_round=1)
    booster.save_model("model.json")
    # fixme: load model
    predt_0 = booster.inplace_predict(df)
    df["c"] = df.c.cat.set_categories(["cdef", "abc"])

    predt_1 = booster.inplace_predict(df)
    if device == "cpu":
        np.testing.assert_allclose(predt_0, predt_1)
    else:
        import cupy as cp

        cp.testing.assert_allclose(predt_0, predt_1)


def test_encoder():
    run_encoder_test("cpu")
    run_encoder_test("cuda")


test_encoder()
