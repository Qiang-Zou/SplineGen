import torch
from typing import Tuple, Union, Optional
class AverageMeter(object):
  def __init__(self):
    self.history = []
    self.reset(record=False)

  def reset(
    self,
    record: bool = True
  ):
    if record:
      self.history.append(self.avg)
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(
    self,
    val: Union[float, int],
    n: int = 1
  ):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

def masked_accuracy(
  output: torch.Tensor,
  target: torch.Tensor,
  mask: torch.Tensor
) -> float:
  with torch.no_grad():
    masked_output = torch.masked_select(output, mask)
    masked_target = torch.masked_select(target, mask)
    accuracy = masked_output.eq(masked_target).float().mean()
    return accuracy