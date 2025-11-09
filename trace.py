from graphviz import Digraph
from autograd.engine import Value

def trace(root_node):
    """
    Trace through the computation graph to collect all nodes and edges.
    
    Args:
        root_node: The output Value node to trace backwards from
        
    Returns:
        tuple: (set of all nodes, set of all edges as (parent, child) tuples)
    """
    all_nodes, all_edges = set(), set()
    
    def build(current_node):
        """Recursively build the graph by traversing parent nodes."""
        if current_node not in all_nodes:
            all_nodes.add(current_node)
            # Add edges from each parent to current node
            for parent_node in current_node._prev:
                all_edges.add((parent_node, current_node))
                build(parent_node)  # Recursively process parent
    
    build(root_node)
    return all_nodes, all_edges


def draw_dot(root_node, format='svg', rankdir='LR'):
    """
    Create a visual representation of the computation graph using Graphviz.
    
    Args:
        root_node: The output Value node to visualize
        format: Output format (png, svg, pdf, etc.)
        rankdir: Graph direction - 'LR' (left to right) or 'TB' (top to bottom)
        
    Returns:
        Digraph object that can be rendered or displayed
    """
    assert rankdir in ['LR', 'TB'], "rankdir must be 'LR' or 'TB'"
    
    # Trace the computation graph
    all_nodes, all_edges = trace(root_node)
    
    # Create a new directed graph with specified layout
    graph = Digraph(format=format, graph_attr={'rankdir': rankdir})
    
    # Add a node for each Value in the computation graph
    for node in all_nodes:
        # Create unique identifier for this node
        node_id = str(id(node))
        
        # Display both data value and gradient in the node
        node_label = "{ data %.4f | grad %.4f }" % (node.data, node.grad)
        graph.node(name=node_id, label=node_label, shape='record')
        
        # If this node was created by an operation, add an operation node
        if node._op:
            op_node_id = node_id + node._op
            graph.node(name=op_node_id, label=node._op)
            # Connect operation node to the value node it produced
            graph.edge(op_node_id, node_id)
    
    # Add edges connecting parent nodes to operation nodes
    for parent_node, child_node in all_edges:
        parent_id = str(id(parent_node))
        child_op_id = str(id(child_node)) + child_node._op
        graph.edge(parent_id, child_op_id)
    
    return graph


# ============================================================================
# EXAMPLE 1: Very simple computation graph
# ============================================================================
input_x = Value(1.0)
output_y = (input_x * 2 + 1).relu()  # Compute: relu(1.0 * 2 + 1) = relu(3.0) = 3.0
output_y.backward()  # Compute gradients via backpropagation

# Visualize the computation graph
graph_simple = draw_dot(output_y)


# ============================================================================
# EXAMPLE 2: Complex multi-variable expression
# ============================================================================
# Create multiple input variables
a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)

# Build a complex expression with multiple operations
# Expression: ((a * b + c) ** 2) / (a + b).relu()
numerator = (a * b + c) ** 2  # (-6 + 10)^2 = 16
denominator = (a + b).relu()   # relu(2 + -3) = relu(-1) = 0 (will cause issues!)
# Let's modify to avoid division by zero
denominator = (a + c).relu()   # relu(2 + 10) = 12
result = numerator / denominator  # 16 / 12 = 1.333...

# Add more complexity with another branch
d = Value(4.0)
e = Value(-1.0)
side_branch = (d * e).relu()  # relu(-4) = 0

# Combine everything
final_output = result + side_branch - a  # 1.333... + 0 - 2 = -0.666...

# Compute gradients
final_output.backward()

# Visualize the complex computation graph
graph_complex = draw_dot(final_output)


# ============================================================================
# EXAMPLE 3: 2D Neuron computation graph
# ============================================================================
import random
from autograd import nn

# Set random seed for reproducibility
random.seed(1337)

# Create a neuron that takes 2 inputs
neuron = nn.Neuron(2)

# Create input values
input_values = [Value(1.0), Value(-2.0)]

# Forward pass through the neuron
neuron_output = neuron(input_values)

# Compute gradients
neuron_output.backward()

# Visualize the neuron's computation graph
graph_neuron = draw_dot(neuron_output)

# Display the graph (in Jupyter notebook)
graph_neuron


# ============================================================================
# EXAMPLE 4: Small neural network (MLP)
# ============================================================================
# Set seed for reproducibility
random.seed(42)

# Create a small multi-layer perceptron: 3 inputs -> 4 hidden -> 2 hidden -> 1 output
mlp = nn.MLP(3, [4, 2, 1])

# Create input data
network_inputs = [Value(0.5), Value(-0.8), Value(1.2)]

# Forward pass
network_output = mlp(network_inputs)

# Compute gradients
network_output.backward()

# Visualize the MLP computation graph (this will be large!)
graph_mlp = draw_dot(network_output)

# To save graphs to files (uncomment if Graphviz is installed):
graph_simple.render('simple_example')
graph_complex.render('complex_example')
graph_neuron.render('neuron_example')
graph_mlp.render('mlp_example')