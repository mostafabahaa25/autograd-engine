class Value:
    """
    Represents a scalar value in a computational graph with automatic differentiation support.
    Stores both the data value and its gradient for backpropagation.
    """

    def __init__(self, data, _children=(), _op=''):
        self.data = data  # The actual scalar value
        self.grad = 0  # Gradient of this value (initialized to 0)
        
        # Internal variables for building the autograd computation graph
        self._backward = lambda: None  # Function to compute gradients during backprop
        self._prev = set(_children)  # Set of parent Value nodes in the graph
        self._op = _op  # String representing the operation that created this node

    def __add__(self, other):
        """Addition operation: self + other"""
        # Convert other to Value if it's a plain number
        other = other if isinstance(other, Value) else Value(other)
        result = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # Gradient flows equally to both operands in addition
            self.grad += result.grad
            other.grad += result.grad
        result._backward = _backward

        return result

    def __mul__(self, other):
        """Multiplication operation: self * other"""
        # Convert other to Value if it's a plain number
        other = other if isinstance(other, Value) else Value(other)
        result = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # Gradient of multiplication: d(a*b)/da = b, d(a*b)/db = a
            self.grad += other.data * result.grad
            other.grad += self.data * result.grad
        result._backward = _backward

        return result

    def __pow__(self, other):
        """Power operation: self ** other"""
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        result = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            # Gradient of power: d(x^n)/dx = n * x^(n-1)
            self.grad += (other * self.data**(other-1)) * result.grad
        result._backward = _backward

        return result

    def relu(self):
        """
        ReLU (Rectified Linear Unit) activation function.
        Returns 0 if input is negative, otherwise returns the input value.
        """
        result = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            # Gradient is 1 if output > 0, else 0 (derivative of ReLU)
            self.grad += (result.data > 0) * result.grad
        result._backward = _backward

        return result

    def backward(self):
        """
        Performs backpropagation to compute gradients for all nodes in the graph.
        Uses topological sort to ensure gradients are computed in the correct order.
        """
        # Build topological ordering of all nodes in the computation graph
        topological_order = []
        visited_nodes = set()
        
        def build_topo(node):
            if node not in visited_nodes:
                visited_nodes.add(node)
                for parent in node._prev:
                    build_topo(parent)
                topological_order.append(node)
        
        build_topo(self)

        # Initialize gradient of output node to 1, then backpropagate
        self.grad = 1
        for node in reversed(topological_order):
            node._backward()

    def __neg__(self):
        """Negation: -self"""
        return self * -1

    def __radd__(self, other):
        """Reverse addition: other + self"""
        return self + other

    def __sub__(self, other):
        """Subtraction: self - other"""
        return self + (-other)

    def __rsub__(self, other):
        """Reverse subtraction: other - self"""
        return other + (-self)

    def __rmul__(self, other):
        """Reverse multiplication: other * self"""
        return self * other

    def __truediv__(self, other):
        """Division: self / other"""
        return self * other**-1

    def __rtruediv__(self, other):
        """Reverse division: other / self"""
        return other * self**-1

    def __repr__(self):
        """String representation showing data and gradient values"""
        return f"Value(data={self.data}, grad={self.grad})"