"""Compositional operators used for implementing models.

This is where most of the library APIs are implemented.
All the operators are functors. They are native PyTorch modules and only have a
forward member for function invocations. We follow the principle of functional
programming. The operators don't keep internal states besides the operator
weights. The operator computation and call has no side effects.
"""
