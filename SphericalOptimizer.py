import math
import torch
from torch.optim import Optimizer

# Spherical Optimizer Class
# Uses the first two dimensions as batch information
# Optimizes over the surface of a sphere using the initial radius throughout
#
# Example Usage:
# opt = SphericalOptimizer(torch.optim.SGD, [x], lr=0.01)

class SphericalOptimizer(Optimizer):
    def __init__(self, optimizer, params, **kwargs):
        self.opt = optimizer(params, **kwargs)
        self.params = params
        with torch.no_grad():
            self.radii = {param: (param.pow(2).sum(tuple(range(2,param.ndim)),keepdim=True)+1e-9).sqrt() for param in params}

    @torch.no_grad()
    def step(self, closure=None):
        loss = self.opt.step(closure)
        for param in self.params:
            param.data.div_((param.pow(2).sum(tuple(range(2,param.ndim)),keepdim=True)+1e-9).sqrt())
            param.mul_(self.radii[param])

        return loss


class StepPostProcessOptimizer(Optimizer):
    def __init__(self, optimizer, params, step_postprocess, **kwargs):
        self.opt = optimizer(params, **kwargs)
        self.params = params
        self.step_postprocess = step_postprocess
        with torch.no_grad():
            self.radii = {param: (param.pow(2).sum(tuple(range(2,param.ndim)),keepdim=True)+1e-9).sqrt() for param in params}

    @torch.no_grad()
    def step(self, closure=None):
        loss = self.opt.step(closure)
        self.step_postprocess(self.params)
        for param in self.params:
            param.data.div_((param.pow(2).sum(tuple(range(2,param.ndim)),keepdim=True)+1e-9).sqrt())
            param.mul_(self.radii[param])
                                          
        return loss
