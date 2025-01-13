import os

import pytest
import torch

from machinelearningtemplate.data import corrupt_mnist


@pytest.mark.skipif(not os.path.exists("data/processed/train_images.pt"), reason="Data files not found")
def test_data():
    train, test = corrupt_mnist()
    assert len(train) in [30000, 50000], "Dataset did not have the correct number of training samples"
    assert len(test) == 5000, "Dataset did not have the correct number of test samples"
    for dataset in [train, test]:
        assert isinstance(dataset, torch.utils.data.Dataset), "Dataset is not an instance of torch.utils.data.Dataset"


if __name__ == "__main__":
    test_data()
