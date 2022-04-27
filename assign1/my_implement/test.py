import autodiff as ad
from visualizer import plot_conputation_graph
from executor import Executor, gradients
import numpy as np

    
x1 = ad.Variable(name="x1")
x2 = ad.Variable(name="x2")
# x3 = ad.Variable(name="x3")
y = x1 * x2 + x2

x1_grad, x2_grad = gradients(y, [x1, x2])
plot_conputation_graph([x1_grad, x2_grad])
plot_conputation_graph([y])
executor = Executor([y, x1_grad, x2_grad])

y, y_grad1_val, x2_grad = executor.run({x1: np.array([1, 2, 7, 10]), x2: np.array([2, 2, 2, 2])})
print(y)
print(y_grad1_val)