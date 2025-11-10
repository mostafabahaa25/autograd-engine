# import random
# from engine import Value

# class Module:
#     """
#     Base class for all neural network modules.
#     Provides common functionality for parameter management.
#     """

#     def zero_grad(self):
#         """Reset gradients of all parameters to zero before backpropagation."""
#         for param in self.parameters():
#             param.grad = 0

#     def parameters(self):
#         """
#         Returns list of all trainable parameters in this module.
#         Should be overridden by subclasses.
#         """
#         return []


# class Neuron(Module):
#     """
#     Represents a single neuron with weights, bias, and optional nonlinearity.
#     """

#     def __init__(self, num_inputs, nonlin=True):
#         """
#         Initialize a neuron with random weights.
        
#         Args:
#             num_inputs: Number of input connections to this neuron
#             nonlin: If True, applies ReLU activation; otherwise linear
#         """
#         # Initialize weights randomly between -1 and 1
#         self.weights = [Value(random.uniform(-1, 1)) for _ in range(num_inputs)]
#         self.bias = Value(0)  # Initialize bias to 0
#         self.nonlin = nonlin  # Whether to apply ReLU activation

#     def __call__(self, inputs):
#         """
#         Forward pass: compute weighted sum of inputs plus bias.
        
#         Args:
#             inputs: List of input values
            
#         Returns:
#             Output value (with ReLU if nonlin=True, otherwise linear)
#         """
#         # Compute: activation = sum(weight_i * input_i) + bias
#         activation = sum((weight * input_val for weight, input_val in zip(self.weights, inputs)), self.bias)
#         return activation.relu() if self.nonlin else activation

#     def parameters(self):
#         """Returns all parameters (weights + bias) of this neuron."""
#         return self.weights + [self.bias]

#     def __repr__(self):
#         """String representation showing neuron type and number of inputs."""
#         neuron_type = 'ReLU' if self.nonlin else 'Linear'
#         return f"{neuron_type}Neuron({len(self.weights)})"


# class Layer(Module):
#     """
#     Represents a layer of neurons (fully connected/dense layer).
#     """

#     def __init__(self, num_inputs, num_outputs, **kwargs):
#         """
#         Initialize a layer with multiple neurons.
        
#         Args:
#             num_inputs: Number of inputs to each neuron in this layer
#             num_outputs: Number of neurons in this layer
#             **kwargs: Additional arguments passed to each Neuron (e.g., nonlin)
#         """
#         self.neurons = [Neuron(num_inputs, **kwargs) for _ in range(num_outputs)]

#     def __call__(self, inputs):
#         """
#         Forward pass: compute output of all neurons in this layer.
        
#         Args:
#             inputs: List of input values
            
#         Returns:
#             Single value if layer has 1 neuron, otherwise list of outputs
#         """
#         outputs = [neuron(inputs) for neuron in self.neurons]
#         return outputs[0] if len(outputs) == 1 else outputs

#     def parameters(self):
#         """Returns all parameters from all neurons in this layer."""
#         return [param for neuron in self.neurons for param in neuron.parameters()]

#     def __repr__(self):
#         """String representation showing all neurons in this layer."""
#         neurons_str = ', '.join(str(neuron) for neuron in self.neurons)
#         return f"Layer of [{neurons_str}]"


# class MLP(Module):
#     """
#     Multi-Layer Perceptron: a feedforward neural network with multiple layers.
#     """

#     def __init__(self, num_inputs, layer_sizes):
#         """
#         Initialize an MLP with specified architecture.
        
#         Args:
#             num_inputs: Number of input features
#             layer_sizes: List of integers specifying the number of neurons in each layer
#                         Example: [16, 16, 1] creates 3 layers with 16, 16, and 1 neurons
#         """
#         # Build list of layer sizes: [num_inputs, layer_sizes[0], layer_sizes[1], ...]
#         sizes = [num_inputs] + layer_sizes
        
#         # Create layers; all layers use ReLU except the last one (linear output)
#         self.layers = [
#             Layer(sizes[i], sizes[i+1], nonlin=(i != len(layer_sizes)-1)) 
#             for i in range(len(layer_sizes))
#         ]

#     def __call__(self, inputs):
#         """
#         Forward pass: pass inputs through all layers sequentially.
        
#         Args:
#             inputs: List of input values
            
#         Returns:
#             Output of the final layer
#         """
#         output = inputs
#         for layer in self.layers:
#             output = layer(output)
#         return output

#     def parameters(self):
#         """Returns all parameters from all layers in this MLP."""
#         return [param for layer in self.layers for param in layer.parameters()]

#     def __repr__(self):
#         """String representation showing all layers in this MLP."""
#         layers_str = ', '.join(str(layer) for layer in self.layers)
#         return f"MLP of [{layers_str}]"




import random
from engine import Value

class Module:
    """
    Base class for all neural network modules.
    Provides common functionality for parameter management.
    """

    def zero_grad(self):
        """Reset gradients of all parameters to zero before backpropagation."""
        for param in self.parameters():
            param.grad = 0

    def parameters(self):
        """
        Returns list of all trainable parameters in this module.
        Should be overridden by subclasses.
        """
        return []


class Neuron(Module):
    """
    Represents a single neuron with weights, bias, and optional nonlinearity.
    """

    def __init__(self, num_inputs, nonlin=True):
        """
        Initialize a neuron with random weights.
        
        Args:
            num_inputs: Number of input connections to this neuron
            nonlin: If True, applies ReLU activation; otherwise linear
        """
        # Initialize weights randomly between -1 and 1
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(num_inputs)]
        self.bias = Value(0)  # Initialize bias to 0
        self.nonlin = nonlin  # Whether to apply ReLU activation

    def __call__(self, inputs):
        """
        Forward pass: compute weighted sum of inputs plus bias.
        
        Args:
            inputs: List of input values (can be Value objects or numbers)
            
        Returns:
            Output value (with ReLU if nonlin=True, otherwise linear)
        """
        # Compute: activation = sum(weight_i * input_i) + bias
        activation = sum((weight * input_val for weight, input_val in zip(self.weights, inputs)), self.bias)
        return activation.relu() if self.nonlin else activation

    def parameters(self):
        """Returns all parameters (weights + bias) of this neuron."""
        return self.weights + [self.bias]

    def __repr__(self):
        """String representation showing neuron type and number of inputs."""
        neuron_type = 'ReLU' if self.nonlin else 'Linear'
        return f"{neuron_type}Neuron({len(self.weights)})"


class Layer(Module):
    """
    Represents a layer of neurons (fully connected/dense layer).
    """

    def __init__(self, num_inputs, num_outputs, **kwargs):
        """
        Initialize a layer with multiple neurons.
        
        Args:
            num_inputs: Number of inputs to each neuron in this layer
            num_outputs: Number of neurons in this layer
            **kwargs: Additional arguments passed to each Neuron (e.g., nonlin)
        """
        self.neurons = [Neuron(num_inputs, **kwargs) for _ in range(num_outputs)]

    def __call__(self, inputs):
        """
        Forward pass: compute output of all neurons in this layer.
        
        Args:
            inputs: List of input values
            
        Returns:
            List of outputs (always a list for consistency between layers)
        """
        outputs = [neuron(inputs) for neuron in self.neurons]
        return outputs

    def parameters(self):
        """Returns all parameters from all neurons in this layer."""
        return [param for neuron in self.neurons for param in neuron.parameters()]

    def __repr__(self):
        """String representation showing all neurons in this layer."""
        neurons_str = ', '.join(str(neuron) for neuron in self.neurons)
        return f"Layer of [{neurons_str}]"


class MLP(Module):
    """
    Multi-Layer Perceptron: a feedforward neural network with multiple layers.
    """

    def __init__(self, num_inputs, layer_sizes):
        """
        Initialize an MLP with specified architecture.
        
        Args:
            num_inputs: Number of input features
            layer_sizes: List of integers specifying the number of neurons in each layer
                        Example: [16, 16, 1] creates 3 layers with 16, 16, and 1 neurons
        """
        # Build list of layer sizes: [num_inputs, layer_sizes[0], layer_sizes[1], ...]
        sizes = [num_inputs] + layer_sizes
        
        # Create layers; all layers use ReLU except the last one (linear output)
        self.layers = [
            Layer(sizes[i], sizes[i+1], nonlin=(i != len(layer_sizes)-1)) 
            for i in range(len(layer_sizes))
        ]

def __call__(self, inputs):
    """
    Forward pass: compute output of all neurons in this layer.
    
    Args:
        inputs: List of input values
        
    Returns:
        List of outputs (always a list for consistency between layers)
    """
    outputs = [neuron(inputs) for neuron in self.neurons]
    return outputs  # This should NOT have any if statement

    def parameters(self):
        """Returns all parameters from all layers in this MLP."""
        return [param for layer in self.layers for param in layer.parameters()]

    def __repr__(self):
        """String representation showing all layers in this MLP."""
        layers_str = ', '.join(str(layer) for layer in self.layers)
        return f"MLP of [{layers_str}]"