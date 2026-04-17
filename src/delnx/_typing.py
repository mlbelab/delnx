from typing import Literal

DataType = Literal["counts", "lognorm", "binary", "scaled", "auto"]
Mode = Literal["sum", "mean"]
ComparisonMode = Literal["all_vs_ref", "all_vs_all", "1_vs_1", "continuous"]
