import os

import xgboost as xgb
from xgboost.testing.external_mem import run_external_memory, run_over_subscription


def main() -> None:
    data_dir = "./data"
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    n = 2**26 + 2**24
    n_batches = 16
    run_external_memory(
        data_dir,
        reuse=False,
        on_host=True,
        n_batches=n_batches,
        n_samples_per_batch=n // n_batches,
    )
    # run_over_subscription(data_dir, True, n_bins=256, n_samples=n, is_sam=True)


if __name__ == "__main__":
    with xgb.config_context(verbosity=3):
        main()
