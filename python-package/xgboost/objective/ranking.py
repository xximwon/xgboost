from typing import Any, Dict, Optional

from xgboost._typing import FloatCompatible

from .base import _BuiltinObjFunction, objective_doc


@objective_doc
class LambdaMartNDCG(_BuiltinObjFunction):
    # fixme: maybe sequence of truncations?
    def __init__(
        self, truncation: FloatCompatible, unbiased: Optional[bool] = None
    ) -> None:
        self.truncation = truncation
        self.unbiased = unbiased

    @staticmethod
    def name() -> str:
        return "lambdamart:ndcg"

    def _save_config(self) -> Dict[str, Any]:
        return {
            "ndcg_param": {
                "lambdamart_truncation": self.truncation,
                "lambdamart_unbiased": self.unbiased,
            }
        }
