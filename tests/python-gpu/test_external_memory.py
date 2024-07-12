import os

from xgboost.testing.external_mem import run_external_memory


def main() -> None:
    data_dir = "./data"
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    run_external_memory(data_dir, True)


if __name__ == "__main__":
    main()
