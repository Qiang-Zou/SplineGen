class AverageMeter(object):
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