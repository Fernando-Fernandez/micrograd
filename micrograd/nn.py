import random
from micrograd.engine import Value

# EXAMPLE:
# from micrograd import nn
# n = nn.Neuron(2)
# x = [Value(1.0), Value(-2.0)]
# y = n(x)

class Module:
    # abstract base class for all neural network components
    # zero_grad() resets all gradients to zero
    # parameters() is overridden by subclasses to provide actual parameters

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):
    # represents a single neuron with learnable weights and bias
    # during initialization, creates random weights between -1 and 1 using Value objects
    # the Value class enables automatic differentiation
    # the nonlin parameter controls whether ReLU activation is applied instead of linear

    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    # the neuron's forward pass computes the weighted sum of inputs plus bias: sum(wi * xi) + b then applies ReLu if enabled
    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):
    # a collection of neurons that operate in parallel on the same input
    # each neuron in the layer receives the full input vector and produces one output value

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        # when the layer contains only one neuron, it returns the single output directly rather than a list
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):
    # MLP = multi layer perceptron
    # the complete feedforward neural network stacking multiple layers
    # accepts the number of input features and a list specifying the size of each hidden layer plus the output layer

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        # only the last layer uses linear activation, that is standard for regression tasks and classification logits
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
