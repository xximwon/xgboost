class Objective:
    def __init__(self) -> None:
        pass


class SoftProb(Objective):
    def __init__(self, num_class: int) -> None:
        self._n_classes = num_class

    def save_config(self) -> dict:
        config = {
            "name": "multi:softprob",
            "softmax_multiclass_param": {"num_class": self._n_classes},
        }
        return config
