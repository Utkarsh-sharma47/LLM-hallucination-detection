"""Dataset schema, split checks, and offline loaders."""

from glassbox.data.loading import load_examples
from glassbox.data.schema import Label, QAExample, make_example_id
from glassbox.data.splits import assert_disjoint_ids

__all__ = [
    "Label",
    "QAExample",
    "assert_disjoint_ids",
    "load_examples",
    "make_example_id",
]
