"""The data package defines the full data pipeline.

We provide dataset implementations in the `datasets` submodule that return a
common data format `DictData`. This data format is used by the pre-processing
functions in the submodule `transforms`. The preprocessing functions are
composed with the datasets in `DataPipe`. Optionally, a reference view sampler
can be added here. The `DataPipe` is input to `torch.data.DataLoader`, for
which we provide utility functions for instantiation that handle also
batch-wise preprocessing and batch collation.
"""
