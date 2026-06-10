import math
import torch

def tolerance_torch(
    x: torch.Tensor,
    bounds: tuple[float, float] = (0.0, 0.0),
    margin: float = 0.0,
    sigmoid: str = "gaussian",
    value_at_margin: float = 0.1,
) -> torch.Tensor:
    """
    PyTorch 版本的 Tolerance 奖励函数，支持 GPU 并行运算。
    """
    lower, upper = bounds
    if lower > upper:
        raise ValueError("lower bound must be less than upper bound")
    if margin < 0:
        raise ValueError("margin must be non-negative")

    # 检查 x 是否落在指定的 [lower, upper] 闭区间内
    in_bounds = torch.logical_and(lower <= x, x <= upper)
    
    if margin == 0.0:
        return torch.where(in_bounds, torch.ones_like(x), torch.zeros_like(x))

    # 计算超出边界的距离并根据 margin 归一化
    d = torch.where(x < lower, lower - x, x - upper) / margin

    # Sigmoid 平滑计算
    if sigmoid == "gaussian":
        scale = math.sqrt(-2 * math.log(value_at_margin))
        v = torch.exp(-0.5 * (d * scale) ** 2)
    elif sigmoid == "long_tail":
        scale = math.sqrt(1 / value_at_margin - 1)
        v = 1 / ((d * scale) ** 2 + 1)
    elif sigmoid == "tanh_squared":
        scale = math.atanh(math.sqrt(1 - value_at_margin))
        v = 1 - torch.tanh(d * scale) ** 2
    elif sigmoid == "hyperbolic":
        scale = math.acosh(1 / value_at_margin)
        v = 1 / torch.cosh(d * scale)
    elif sigmoid == "linear":
        scale = 1 - value_at_margin
        scaled_d = d * scale
        v = torch.where(torch.abs(scaled_d) < 1, 1 - scaled_d, torch.zeros_like(x))
    elif sigmoid == "quadratic":
        scale = math.sqrt(1 - value_at_margin)
        scaled_d = d * scale
        v = torch.where(torch.abs(scaled_d) < 1, 1 - scaled_d**2, torch.zeros_like(x))
    else:
        raise ValueError(f"Unknown sigmoid type {sigmoid!r}.")

    # 在边界内直接给 1.0，边界外按衰减曲线给分
    return torch.where(in_bounds, torch.ones_like(x), v)