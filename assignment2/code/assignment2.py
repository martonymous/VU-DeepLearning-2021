import vugrad as vg
import numpy as np

a = vg.TensorNode(np.random.randn(1, 1))
b = vg.TensorNode(np.random.randn(1, 1))

c = a + b

print("c.value:\n", c.value)
print("c.source:\n", c.source)
print("c.source.inputs:\n", c.source.inputs[0])
print("a.grad:\n", a.grad)


