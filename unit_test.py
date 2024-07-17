import torch
import torch.nn.functional as F
from torchmetrics import Metric
from torch.testing import assert_allclose

# Assuming your classes SumExceptBatchKL and SumExceptBatchMetric are defined here

def test_sum_except_batch_kl():
    # Create instances of the metric
    metric = SumExceptBatchKL()

    # Generate sample data for p and q
    p = torch.randn(10, 5)  # 10 samples, 5 features each
    p = F.softmax(p, dim=1)  # Convert to probabilities
    q = torch.randn(10, 5)
    q = F.softmax(q, dim=1)

    # Update the metric with the generated data
    metric.update(p, q)

    # Calculate expected KL divergence using PyTorch
    expected_kl = F.kl_div(q.log(), p, reduction='batchmean') * p.size(0)

    # Compute metric's result
    result = metric.compute()

    # Check if they are close
    assert_allclose(result, expected_kl, atol=1e-5)

def test_sum_except_batch_metric():
    # Create instances of the metric
    metric = SumExceptBatchMetric()

    # Generate sample data
    values = torch.randn(10, 5)  # 10 samples, 5 features each

    # Update the metric with the generated data
    metric.update(values)

    # Calculate expected average
    expected_average = values.mean()

    # Compute metric's result
    result = metric.compute()

    # Check if they are close
    assert_allclose(result, expected_average, atol=1e-5)

# Run tests
test_sum_except_batch_kl()
test_sum_except_batch_metric()
