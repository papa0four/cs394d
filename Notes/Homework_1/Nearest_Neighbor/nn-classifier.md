## Nearest Neighbor Classifier Notes

### Make Data Explained

```python3
@classmethod
def make_data(cls, x: list[list[float]], y: list[float]) -> tuple(torch.Tensor, torch.Tensor):
    """
    Warmup: Convert the data into PyTorch tensors
    Assumptios:
    - len(x) == len(y)

    Args:
        x: list of lists of floats, data
        y: list of floats, labels

    Returns:
        tuple of x and y both torch.Tensor's.
    """
    x_tensor = torch.as_tensor(x, dtype=torch.float32)
    y_tensor = torch.as_tensor(y, dtype=torch.float32)
    return x_tensor, y_tensor
```

* `torch.as_tensor(x, dtype=torch.float32)` directly converts the list of lists `x` to a 2D tensor of type float32 without creating an intermediate tensor.
* `torch.as_tensor(y, dtype=torch.float32)` directly converts the list `y` to a 1D tensor of type float32 without creating an intermediate tensor.

### Compute Data Statistics Explained

```python3
@classmethod
def compute_data_statistics(cls, x: torch.Tensor) -> tuple[torch.Tensor]:
    """
    Compute the mean and standard deviation of the data.
    Each row denotes a single data point.

    Args:
        x: 2D tensor data shape = [N, D]

    Returns:
        tuple of mean and standard deviation of the data.
        Both should have a shape [1, D]
    """
    mean = torch.mean(x, dim=0, keepdim=True)
    std = torch.std(x, dim=0, keepdim=True)
    return mean, std
```

* `torch.mean(x, dim=0, keepdim=True)` computes the mean of the data along the rows (i.e., for each column) and retains the dimension to ensure the resulting tensor has shape `[1, D]`.
* `torch.std(x, dim=0, keepdim=True)` computes the standard deviation along the rows and retains the dimension to ensure the resulting tensor has shape `[1, D]`
* The method returns a tuple containing the mean and standard deviation tensors

### Input Normalization Explained

```python3
def input_normalization(self, x: torch.Tensor) -> torch.Tensor:
    """
    Normalize the input x using the mean and std computed from the data in __init__

    Args:
        s: 1D or 2D tensor shape = [D] or [N, D]

    Returns:
        normalized 2D tensor shape = x.shape
    """
    epsilon = 1e-8 # to prevent division by 0 edge case
    return (x - self.data_mean) / (self.data_std + epsilon)
```

* **Element-wise Opertaions**: PyTorch supports broadcasting for element-wise operations, so subtracting `self.data_mean` and dividing by `self.data_std` will be applied correctly to each element of `x`, whether `x` is 1D or 2D.
* **Shape Consistency**: The resulting tensor will have the same shape as the input `x` because the normalization is applied element-wise.
* The value of `epsilon` (`1e-8`) is chosen to be small enough so that it doesn't significantly alter the standard deviation values but large enough to prevent division by zero.
* We directly perform the normalization and add `epsilon` to `self.data_std` in the return statement leveraging PyTorch's broadcasting capabilities.

### Get Nearest Neighbor Explained

```python3
def get_nearest_neighbor(self, x: torch.Tensor) -> tuple[torch.Tensor]:
    """
    Find the input x's nearest neighbor and the corresponding label.

    Args:
        x: 1D tensor shape = [D]

    Returns:
        tuple of the nearest neighbor data point [D] and its label [1]
    """
    # Normalize the input tensor
    x = self.input_normalization(x)

    # Compute the Euclidean distance and find the index of the nearest neighbor
    idx = torch.argmin(torch.norm(self.data_normalized - x, dim=1))
    
    # Return the nearest neighbor data point and its label
    return self.data[idx], self.label[idx]
```

* `self.input_normalization(x)` normalizes the input tensor `x`
* `torch.norm(self.data_normalized - x, dim=1)` computes the Euclidean distance between `x` and each data point in the normalized dataset
* `torch.argmin()`finds the index of the minimum distance
* `self.data[idx]` retrieves the nearest neighbor data point
* `self.label[idx]` retrieves the label corresponding to the nearest neighbor data point

### Get K Nearest Neighbor Explained

```python3
def get_k_nearest_neighbor(self, x: torch.Tensor, k: int) -> tuple[torch.Tensor]:
    """
    Find the k-nearest neighbors of input x from the data

    Args:
        x: 1D tensor shape = [D]
        k: int, number of neighbors

    Returns:
        tuple of the k-nearest neighbors data points and their labels
        data points will be size (k, D)
        labels will be size (k,)
    """
    # Normalize the input tensor
    x = self.input_normalization(x)

    # Compute the Euclidean distance and find the indices of the k nearest neighbors
    _, idx = torck.topk(torch.norm(self.data_normalized - x, dim=1), k, largest=False)

    # Return the k-nearest neighbor data points and their labels
    return self.data[idx], self.label[idx]
```

* `self.input_normalization(x)` normalizes the input tensor `x`
* `torch.norm(self.data_normalized - x, dim=1)` computes the Euclidean distance between `x` and each data point in the normalized dataset
* `torch.topk(..., k, largest=False)` finds the indices of the `k` smallest distances (nearest neighbors)
* `self.data[idx]` retrieves the `k` nearest neighbor data points
* `self.label[idx]` retrieves the labels corresponding to the `k` nearest neighbor data points

### K Nearest Neighbor Regression Explained

```python3
def knn_regression(self, x: torch.Tensor, k: int) -> torch.Tensor:
    """
    Use the k-nearest neighbors of the input x to predict its regression label.
    The prediction will be the average value of the labels from the k neighbors.

    Args:
        x: 1D tensor [D]
        k: int, number of neighbors

    Returns:
        average value of labels from the k neighbors. Tensor of shape [1]
    """
    # Normalize the input tensor
    x = self.input_normalization(x)

    # Get the k-nearest neighbors and their labels
    _, labels = self.get_k_nearest_neighbor(x, k)

    # Compute the average value of the labels
    average_labels = torch.mean(labels)

    return average_label
```

* `self.input_normalization(x)` normalizes the input tensor `x`
* `self.get_k_nearest_neighbor(x, k)` finds the k-nearest neighbors and their labels
* `torch.mean(labels)` computes the average value of the labels of the k-nearest neighbors
* The method returns the average label, which is the predicted regression label