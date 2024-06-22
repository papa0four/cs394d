## PyTorch Basics

### Make it PyTorch 1 Explained

```python3
def make_it_pytorch_1(x: torch.Tensor) -> torch.Tensor:
    """
    Return every 3rd element of the input tensor.

    x is a 1D tensor

    --------
    y = []
    for i, v in enumerate(x):
        if i % 3 == 0:
            y.append(v)
    return torch.stack(y, dim=0)
    --------

    Solution length: 13 characters
    """
    return x[::3]
```
* The above return statement is using slicing to select every 3rd element from the input tensor `x`, which is efficient and concise.

### Make it PyTorch 2 Explained

```python3
def make_it_pytorch_2(x: torch.Tensor) -> torch.Tensor:
    """
    Return the maximum value of each row of the final dimension of the input tensor

    x is a 3D tensor

    --------
    n, m, _ = x.shape
    y = torch.zeros(n, m)
    for i in range(n):
        for j in range(m):
            maxval = float("-inf")
            for v in x[i, j]:
                if v > maxval:
                    maxval = v
            y[i, j] = maxval
    return y
    --------

    Solution length: 26 characters
    """
    return torch.max(x, dim=-1).values
```

* The `torch.max` function is used with the `dim=-1` argument to find the maximum value along the last dimension of the tensor.
* The `.values` attribute is used to get the maximum values themselves (as `torch.max` returns a namedtuples with values and indices).

### Make it PyTorch 3 Explained

```python3
def make_it_pytorch_3(x: torch.Tensor) -> torch.Tensor:
    """
    Return the unique elements of the input tensor in sorted order

    x can have any dimension

    --------
    y = []
    for i in x.flatten():
        if i not in y:
            y.append(i)
    return torch.as_tensor(sorted(y))
    --------

    Solution length: 22 characters
    """
    return torch.unique(x, sorted=True)
```

* The `torch.unique` function is used to find the unique elements of the input tensor.
* The `sorted=True` argument ensures that the unique elements are returned in sorted order

### Make it PyTorch 4 Explained

```python3
def make_it_pytorch_4(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Return the number of elements in y that are greater than the mean of x

    x and y can have any shape

    --------
    a = 0
    b = 0
    for in x.flatten():
        a += 1
        b += 1
    mean = a / b
    c = 0
    for in y.flatten():
        if i > mean:
            c += 1
    return torch.as_tensor(c)
    --------

    Solution lenght: 27 characters
    """
    return (y > torch.mean(x)).sum()
```

* `torch.mean(x)` computes the mean of the tensor `x`
* `y > torch.mean(x)` creates a boolean tensor where each element indicates whether the corresponding element in `y` is greater than the mean of `x`
* `.sum()` counts the number of `True` values in the boolean tensor, which corresponds to the number of elements in `y` that are greater than the mean of `x`

### Make it PyTorch 5 Explained

```python3
def make_it_pytorch_5(x: torch.Tensor) -> torch.Tensor:
    """
    Return the transpose of the input tensor

    x is a 2D tensor

    --------
    y = torch.zeros(x.shape[1], x.shape[0])
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            y[j, i] = x[i, j]
    return y
    --------

    Solution length: 11 characters
    """
    return x.t()
```

* `x.t()` is a shorthand for `torch.transpose(x, 0, 1)`, which transposes the 2D tensor by swapping its dimensions

### Make it PyTorch 6 Explained

```python3
def make_it_pytorch_6(x: torch.Tensor) -> torch.Tensor:
    """
    Return the diagonal elements of the input tensor

    x is a 2D tensor

    --------
    y = []
    for i in range(min(x.shape[0], x.shape[1])):
        y.append(x[i, i])
    return torch.as_tensor(y)
    --------

    Solution length: 19 characters
    """
    return x.diagonal()
```

* `x.diagonal()` returns the diagonal elements of the 2D tensor `x`

### Make it PyTorch 7 Explained

```python3
def make_it_pytorch_7(x: torch.Tensor) -> torch.Tensor:
    """
    Return the diagonal elements of the input tensor in reverse order

    x is a 2D tensor
    
    --------
    y = []
    for i in range(min(x.shape[0], x.shape[1])):
        y.append(x[i, x.shape[1] - i - 1])
    return torch.as_tensor(y)
    --------

    Solution length: 27 characters
    """
    return x.flip(1).diagonal()
```

* `x.flip(1)` flips the tensor along the columns, effectively reversing each row
* `.diagonal()` then extracts the main diagonal of the flipped tensor, which corresponds to the original anti-diagonal elements in reverse order

### Make it PyTorch 8 Explained

```python3
def make_it_pytorch_8(x: torch.Tesnor) -> torch.Tensor:
    """
    Return the cumulative sum of the input tensor

    x is a 1D tensor

    --------
    if len(x) == 0:
        return torch.as_tensor(x)
    y = [x[0]]
    for i in range(1, len(x)):
        y.append(y[i - 1] + x[i])
    return torch.as_tensor(y)
    --------

    Solution length: 22 characters
    """
    return torch.cumsum(x, dim=0)
```

* `torch.cumsum(x, dim=0)` computes the cumulative sum of the elements of `x` along dimension 0

### Make it PyTorch 9 Explained

```python3
def make_it_pytorch_9(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the sum of all elements in the rectangle up to (i, j)th element

    x is a 2D tensor

    --------
    y = torch.zeros_like(x)
    for i in range(0, x.shape[0]):
        for j in range(0, x.shape[1]):
            y[i, j] = x[i, j]
            if i > 0:
                y[i, j] += y[i - 1, j]
            if j > 0:
                y[i, j] += y[i, j - 1]
            if i > 0 and j > 0:
                y[i, j] -= y[i - 1, j - 1]
    return y
    --------

    Solution length: 36 characters
    """
    return torch.cumsum(torch.cumsum(x, dim=0), dim=1)
```

* `torch.cumsum(x, dim=0)` computes the cumulative sum along the rows
* Applying inner `torch.cumsum()` to the result computes the cumulative sum along the columns, resulting in the desired 2D cumulative sum array

### Make it PyTorch 10 Explained

```python3
def make_it_pytorch_10(x: tprch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """
    Return the input tensor with all elements less than c set to 0

    x is a 2D tensor
    c is a scalar tensor (dimension 0)

    --------
    y = torch.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i, j] < c:
                y[i, j] = 0.0
            else:
                y[i, j] = x[i, j]
    return y
    --------

    Solution length: 49 characters
    """
    return torch.where(x < c, torch.zeros_like(x), x)
```

* `torch.where(x < c, torch.zeros_like(x), x)` creates a tensor where each element is 0 if the corresponding element in `x` is less than `c`, otherwise it retains the value from `x`

### Make it PyTorch 11 Explained

```python3
def make_it_pytorch_11(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """
    Return the row and column indices of the elements less than c

    x is a 2D tensor
    c is a scalar tensor (dimension 0)

    The output is a 2 x n tensor where n is the number of elements less than c

    --------
    row, col = [], []
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i, j] < c:
                row.append(i)
                col.append(j)
    return torch.as_tensor([row, col])
    --------

    Solution length: 30 characters
    """
    return torch.nonzero(x < c).t()
```

* `x < c` creates a boolean tensor where each element indicates whether the corresponsing element in `x` is less than `c`
* `torch.nonzero(x < c).t()` returns a tensor with the indices of the non-zero (True) elements
* `.t()` transposes the resulting tensor to get the shape (2, n)

### Make it PyTorch 12 Explained

```python3
def make_it_pytorch_12(x: torch.Tesnor, m: torch.BoolTensor) -> torch.Tensor:
    """
    Return the elements of x where m is True

    x and m are 2D tensors

    --------
    y = []
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if m[i, j]:
                y.append(x[i, j])
    return torch.as_tensor(y)
    --------

    Solution length: 11 characters
    """
    return torch.masked_select(x, m)
```

* `torch.masked_select(x, m)` returns a 1D tensor containing the elements of `x` where the corresponding elements of `m` are `True`

### Make it PyTorch 1 Extra Explained

```python3
def make_it_pytorch_extra_1(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Return the difference between consecutive elements of the sequence [x, y]

    x and y are 1D tensors

    --------
    xy = []
    for xi in x:
        xy.append(xi)
    for yi in y:
        xy.append(yi)

    z = []
    for xy1, xy2 in zip(xy[1:], xy[:-1]):
        z.append(xy1 - xy2)
    return torch.as_tensor(z)
    --------

    Solution length: 36 characters
    """
    return torch.diff(torch.cat((x, y)))
```

* `torch.cat((x, y))` concatenates the tensors `x` and `y` along the specified dimension (default 0 for 1D tensors)
* `torch.diff` computes the difference between consecutive elements of the concatenated tensor

### Make it PyTorch Extra 2 Explained

```python3
def make_it_pytorch_extra_2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Find the number of elements in x that are equal to (abs(x_i-y_j) < 1e-3) to at least one element in y

    x and y are 1D tensors

    --------
    count = 0
    for xi in x:
        for yi in y:
            if (abs(xi - yi) < 1e-3):
                count += 1
                break
    return torch.as_tensor(count)
    --------

    Solution length: 64 characters
    """
    return ((x[:, None] - y).abs() < 1e-3).any(dim=1).sum()
```

* `x[:, None]` reshapes `x` to a 2D tesnor with shape `(len(x), 1)`, enabling broadcasting
* `(x[:, None] - y)` performs element-wise subtraction between each element in `x` and every element in `y`. resulting in a 2D tensor
* `abs()` takes the absolute value