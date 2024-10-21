class AverageMeter(object):
  """
  Computes and stores the average and current value

  Adapted from pointer-networks-pytorch by ast0414:
    https://github.com/ast0414/pointer-networks-pytorch
  """

  def __init__(self):
    self.vals = []
    self.ns = []

  def update(
    self,
    val,
    n
  ):
    self.vals.append(val)
    self.ns.append(n)

  def acc(
    self,
  ):
    s=sum(self.vals)
    num=sum(self.ns)
    return s/num