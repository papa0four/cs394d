## Weather Forecast Assignment Notes

### Find Min and Max Per Day Explained

1. Use `torch.min` to find the minimum temperature for each day
2. Use `torch.max` to find the maximum temperature for each day

```python3
def find_min_and_max_per_day(self) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find the max and min temperatures per day

    Returns:
        min_per_day: tensor of size (num_days,)
        max_per_day: tensor of size (num_days,)
    """
    min_per_day = torch.min(self.data, dim=1).values
    max_per_day = torch.max(self.data, dim=1).values
    return min_per_day, max_per_day
```

* `torch.min(self.data, dim=1).values` computes the minimum value along the second dimension (i.e., for each day). The `values` attribute extracts the minimum values themselves
* `torch.max(self.data, dim=1).values` computes the maximum value along the second dimension (i.e., for each day). The `.values` attribute extracts the maximum values thmeselves

## Find the Largest Drop Explained

1. Compute the average temperature for each day
2. Compute the difference in average temperatures between consecutive days
3. Find the largest negative difference

```python3
def find_the_largest_drop(self) -> torch.Tensor:
    """
    Find the largest drop in average temperature.
    This should be a negative number.

    Returns:
        tensor of a single value, the difference in temperature
    """
    return torch.min(torch.diff(torch.mean(self.data, dim=1)))
```

* `torch.mean(self.data, dim=1)` computes the average temperature for each day
* `torch.diff()` computes the differences between consecutive days
* `torch.min()` finds the largest drop, which is the minimum value among the differences

### Find the Most Extreme Day Explained

1. Compute the average temperature for each day
2. Compute the absolute differences between each measurement and the day's average temperature
3. Find the index of the maximum absolute difference for each day
4. Use these indices to select the corresponding temperature measurements from the original data

```python3
def find_the_most_extreme_day(self) -> torch.Tensor:
    """
    For each day, find the measurement that differs the most from the day's average temperature

    Returns:
        tensor with size (num_days,)
    """
    abs_diff = torch.abs(self.data - (torch.mean(self.data, dim=1, keepdim=True)))
    max_diff_indices = torch.argmax(abs_diff, dim=1)
    most_extreme_values = self.data[torch.arange(self.data.size(0)), max_diff_indices]
    return most_extreme_values
```

* `torch.mean(self.data, dim=1, keepdim=True)` computes the average temperature for each day and retains the dimensions for broadcasting
* `torch.abs(self.data - (torch.mean(..., ..., ...)))` computes the absolute differences between each measurement and the day's average temperature
* `torch.argmax(torch.abs(), dim=1)` finds the index of the maximum absolute difference for each day
* `self.data[torch.arange(self.data.size(0)), max_diff_indices]` selects the temperature measurements from the original data using the computed indices

### Max Last K Days Explained

1. Selecting the data for the last `k` days
2. Finding the maximum temperature for each of these days

```python3
def max_last_k_days(self, k: int) -> torch.Tensor:
    """
    Find the maximum temperature over the last k days

    Returns:
        tensor of size (k,)
    """
    return torch.max(self.data[-k:], dim=1).values
```

* `self.data[-k:]` selects the data for the last `k` days
* `torch.amx(self.data[-k:], dim=1).values` computes the maximum temperature for each day in the selected data.

### Predict Temperature Explained

1. Selecting the data for the last `k` days
2. Computing the average temperature over these `k` days

```python3
def predict_temperature(self, k: int) -> torch.Tensor:
    """
    From the dataset, predict the temperature of the next day.
    The prediction will be the average of the temperatures over the past k days.

    Args:
        k: int, number of days to consider

    Returns:
        tensor of a single value, the predicted temperature
    """
    return torch.mean(self.data[-k:])
```

* `self.data[-k:]` selects the data for the last `k` days
* `torch.mean(self.data[-k:])` computes the average temperature over these `k` days

### What Day Is This From Explained

1. Compute the sum of absolute differences between the given temperature measurements and each day's measurements in the dataset
2. Find the index of the day with the smallest sum of absolute differences

```python3
def what_day_is_this_from(self, t: torch.FloatTensor) -> torch.LongTensor:
    """
    You go on a stroll next to the weather station, where this data was collected.
    You find a phone with severe weather damage.
    The only thing that you can see in the screen is the temperature reading of one full day, right before it broke.

    You want to figure out what day it broke.

    The dataset we have starts from Monday.
    Given a list of 10 temperature measurements, find the day in a week that the temperature is most likely measured on.

    We measure the difference using 'sum of absolute difference per measurement:
        d = |x1 - t1| + |x2 - t2| + ... + |x10 - t10|

    Args:
        t: tensor of size (10,), temperature measurements

    Returns:
        tensor of a single value, the index of the closest data element
    """
    differences = torch.sum(torch.abs(self.data - t), dim=1)
    return torch.argmin(differences)
```

* `torch.abs(self.data - t)` computes the absolute differences between the given measurements `t` and each day's measurements in the dataset
* `torch.sum(..., dim=1)` sums the absolute differences for each day
* `torch.argmin(differences)` finds the index of the day with the smallest sum of absolute differences
