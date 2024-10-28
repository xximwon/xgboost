from xgboost import testing as tm
from distributed import get_client


def make_ranking(n_samples: int, n_features: int, n_query_groups=3, max_rel=3):
    with get_client() as client:
        with client.as_current() as client:
            client.submit(tm.make_ltr)
