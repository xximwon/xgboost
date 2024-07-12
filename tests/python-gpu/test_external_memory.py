import os

from xgboost.testing.external_mem import run_external_memory, run_over_subscription


def main() -> None:
    data_dir = "./data"
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    # run_external_memory(data_dir, True, n_samples_per_batch=2**16)
    run_over_subscription(data_dir, True, n_samples=2**16)


if __name__ == "__main__":
    main()
