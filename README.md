# Autograd: A Tiny Autograd Engine

is a minimal automatic differentiation engine and neural network framework implemented from first principles in pure Python.
This project serves as a research-oriented exploration of the core algorithms behind modern deep learning libraries, focusing on reverse-mode autodiff, computational graph construction, and gradient-based optimization.

The implementation aims to provide a transparent and educational perspective on how neural networks learn through backpropagation, offering an interpretable foundation for further experimentation in AI systems design and computational graph theory.
This is an educational implementation that shows how neural networks and backpropagation work under the hood, operating on individual scalars rather than tensors.

## Table of contents

- [Project Structure](#Project-Structure)
- [Getting Started](#Getting-Started)
  - [Prerequisites](#Prerequisites)
  - [Basic Usage](#Basic-Usage)
  - [1. Simple Scalar Operations](#1.-Simple-Scalar-Operations)
  - [2. Building a Neural Network](#2.-Building-a-Neural-Network)
  - [3. Training Example](#3.-Training-Example)
  - [4. Visualizing Computation Graphs](#4.-Visualizing-Computation-Graphs)
- [Core Components](#Core-Components)
  - [Value Class](#Value-Class)
  - [Neural Network Components](#Neural-Network-Components)
  - [Visualization](#Visualization)
- [Examples](#Examples)
- [How It Works](#How-It-Works)
  - [Automatic Differentiation](#Automatic-Differentiation)
  - [Neural Networks](#NeuralNetworks)
- [Educational Value](#Educational-Value)
- [Limitations](#Limitations)
- [Further Reading](#Further-Reading)
- [Contributing](#Contributing)
- [License](#License)
## Project Structure

```
.
├── engine.py          # Core autograd engine with Value class
├── nn.py             # Neural network components (Module, Neuron, Layer, MLP)
├── trace.py          # Visualization tools for computation graphs
└── README.md         # This file
```

## Getting Started

### Prerequisites

```bash
pip install graphviz
```

**Note:** You also need to install Graphviz system binaries:
- **Windows**: Download from [graphviz.org](https://graphviz.org/download/) and add to PATH
- **Mac**: `brew install graphviz`
- **Linux**: `sudo apt-get install graphviz`

### Basic Usage

#### 1. Simple Scalar Operations

```python
from engine import Value

# Create values
a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)

# Build expressions
d = a * b + c      # -6 + 10 = 4
e = d.relu()       # relu(4) = 4

# Compute gradients
e.backward()

print(f"e.data = {e.data}")      # 4.0
print(f"a.grad = {a.grad}")      # Gradient of e with respect to a
```

#### 2. Building a Neural Network

```python
from nn import MLP
from engine import Value

# Create a 3-layer network: 3 inputs -> 4 hidden -> 4 hidden -> 1 output
model = MLP(3, [4, 4, 1])

# Forward pass
inputs = [Value(2.0), Value(3.0), Value(-1.0)]
output = model(inputs)

# Backward pass
output.backward()

# Access parameters
params = model.parameters()
print(f"Total parameters: {len(params)}")
```

#### 3. Training Example

```python
# Training data (XOR problem)
xs = [
    [Value(0.0), Value(0.0)],
    [Value(0.0), Value(1.0)],
    [Value(1.0), Value(0.0)],
    [Value(1.0), Value(1.0)],
]
ys = [Value(0.0), Value(1.0), Value(1.0), Value(0.0)]

# Create model
model = MLP(2, [4, 1])

# Training loop
learning_rate = 0.01
for epoch in range(100):
    # Forward pass
    predictions = [model(x) for x in xs]
    
    # Compute loss (MSE)
    loss = sum((pred - target)**2 for pred, target in zip(predictions, ys))
    
    # Backward pass
    model.zero_grad()
    loss.backward()
    
    # Update parameters
    for param in model.parameters():
        param.data -= learning_rate * param.grad
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.data}")
```

#### 4. Visualizing Computation Graphs

```python
from trace import draw_dot
from engine import Value

# Build an expression
x = Value(2.0)
y = Value(3.0)
z = x * y + x ** 2
z.backward()

# Create visualization
graph = draw_dot(z)

# Display in Jupyter notebook
graph

# Or save to file (requires Graphviz installed)
graph.render('computation_graph', format='png')
```

## Core Components

### Value Class (`engine.py`)

The fundamental building block that wraps scalar values and tracks gradients.

**Supported Operations:**
- Addition: `+`
- Multiplication: `*`
- Power: `**`
- Division: `/`
- Subtraction: `-`
- ReLU activation: `.relu()`
- Backpropagation: `.backward()`

### Neural Network Components (`nn.py`)

#### Module
Base class providing:
- `parameters()`: Returns all trainable parameters
- `zero_grad()`: Resets gradients to zero

#### Neuron
Single neuron with:
- Random weight initialization
- Bias term
- Optional ReLU activation

#### Layer
Fully connected layer with multiple neurons

#### MLP (Multi-Layer Perceptron)
Complete neural network with multiple layers

### Visualization (`trace.py`)

- `trace(root_node)`: Traces computation graph to extract nodes and edges
- `draw_dot(root_node)`: Creates Graphviz visualization of the computation graph

## Examples

See `trace.py` for complete examples:
1. **Simple expression**: Basic arithmetic with ReLU
2. **Complex multi-variable expression**: Multiple operations and branches
3. **Single neuron**: 2D neuron with weights and bias
4. **Small MLP**: Multi-layer network showing full backpropagation

## How It Works

### Automatic Differentiation

Micrograd uses **reverse-mode automatic differentiation**:

1. **Forward Pass**: Operations build a dynamic computation graph
2. **Backward Pass**: Gradients flow backward through the graph using the chain rule

Example:
```python
a = Value(2.0)
b = Value(3.0)
c = a * b  # c = 6.0

c.backward()
# dc/da = b = 3.0
# dc/db = a = 2.0
print(a.grad)  # 3.0
print(b.grad)  # 2.0
```

### Neural Networks

Neural networks are built by:
1. Composing `Value` operations in neurons
2. Stacking neurons into layers
3. Chaining layers together
4. Using `.backward()` to compute all gradients automatically

## Educational Value

This implementation is ideal for:
- Understanding how autograd engines work
- Learning backpropagation from first principles
- Seeing the connection between calculus and neural networks
- Building intuition before using production frameworks

## Limitations

- **Scalar-only**: Works on individual numbers, not tensors/matrices
- **Slow**: Not optimized for performance (educational purpose)
- **No GPU support**: Pure Python implementation
- **Limited operations**: Only basic mathematical operations

For production use, consider: PyTorch, TensorFlow, JAX

## Further Reading

- [Automatic Differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation)
- [Backpropagation](https://en.wikipedia.org/wiki/Backpropagation)
- [Neural Networks](https://en.wikipedia.org/wiki/Artificial_neural_network)

## Contributing

This is an educational project. Feel free to:
- Add more activation functions (sigmoid, tanh, etc.)
- Implement optimizers (SGD with momentum, Adam, etc.)
- Add more layer types (dropout, batch normalization)
- Improve visualization

## License

This project is intended for educational purposes.
