"""
Aim :
    this is our aim, we can write follow code for construct a computation graph.
    Code:
        x1 = ad.Variable(name = "x1").
        x2 = ad.Variable(name = "x2").
        y = x1 * x2 + x1.
    so where is the computation graph store?
        store in the y. this project save the inputs node.
        
        
Some essential knowledge:
    we need code y = x1 * x2 + x1 to construct a computation graph.
    x1 * x2 is call the x1 method of __mul__, x1 + x2 is call the x1 method of __add__.
    so we need return a new node in the method __add__ or __mul__ or other op function in the Node class.
    
"""
import numpy as np
from unicodedata import name

class Node(object):
    """ This is the Node of Computation Graph. """
    def __init__(self):
        """ Constructor, new node is indirectly created by Op object __call__ method
        
        Instance variables
        ------------------
        self.inputs: the list of input nodes
        self.op: the associated op object
            e.g. add_op object if this node is created by adding two other nodes
        self.const_attr: the add or multply constant.
            e.g. self.const_attr = 5 if this node is created by x+5
        self.name: node name for debugging purposes
        """
        self.inputs = []
        self.op = None
        self.const_attr = None
        self.name = ""
    
    def __add__(self, other):
        """ Adding two nodes return a new node """
        if isinstance(other, Node):
            new_node = add_op(self, other)
        else:
            new_node = add_byconst_op(self, other)
        return new_node
    
    def __mul__(self, other):
        """ Multiply two nodes or one node one const return a new node. """
        if isinstance(other, Node):
            new_node = mul_op(self, other)
        else:
            new_node = mul_byconst_op(self, other)
        return new_node
    
    __radd__ = __add__
    __rmul__ = __mul__
    
    def __truediv__(self, other):
        """ divide two nodes return a new node """
        if isinstance(other, Node):
            new_node = div_op(self, other)
        else:
            pass
        return new_node
    
    def __neg__(self,):
        """ negate a node and return a new node. """
        return negate_op(self)
    
    def __sub__(self, other):
        """ sub operation of two ndoes and return a new node. """
        return sub_op(self, other)
        
    def __str__(self):
        return self.name
    
    __repr__ = __str__

class Op(object):
    """ Op represents operations performed on nodes """
    def __call__(self):
        """ Create a new node and associate the op object with the node
        Returns:
            a new node after operation
        """
        new_node = Node()
        new_node.op = self
        return new_node

    def compute(self, node, input_vals):
        """Given values of input nodes, compute the output value.

        Args:
            node (Node): the operation of inputs.
            input_vals (list): the values of inputs.

        Raises:
            An output value of the node.
        """
        raise NotImplementedError
    
    def gradients(self, node, output_grad):
        """Given value of output gradient, compute gradient contributions to each input node.

        Parameters
        ----------
        node: node that performs the gradient.
        output_grad: value of output gradient summed from children nodes' contributions

        Returns
        -------
        A list of gradient contributions to each input node respectively.
        """
        raise NotImplementedError
    
class PlaceholderOp(Op):
    """Op to feed value to a nodes."""
    def __call__(self):
        """Creates a variable node."""
        new_node = Op.__call__(self)
        return new_node
    
    def compute(self, node, input_vals):
        """No compute function since node value is fed directly in Executor."""
        assert False, "placeholder values provided by feed_dict"
        
    def gradients(self, node, output_grad):
        """Return None gradients because there is no input."""
        return None

class AddOp(Op):
    """ AddOp represents element-wise add two nodes """
    def __call__(self, node_A, node_B):
        """
        Args:
            node_A (Node): first add node.
            node_B (Node): second add node.
        Returns:
            a new node after add operation.
        """
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = f"({node_A.name}+{node_B.name})"
        return new_node
    
    def compute(self, node, input_vals):
        """Given two numbers return the element-wise addition of the two numbers.
        Args:
            node (Node): the operation of inputs.
            input_vals (list): the values of inputs.

        Returns:
            the element-wise addition of the two numbers.
        """
        assert len(input_vals) == 2 and len(node.inputs) == 2
        return input_vals[0] + input_vals[1]
    
    def gradients(self, node, output_grad):
        """ compute the output gradient contribute to each input node.
        Args:
            node (Node): the operation node in the computation graph.
            output_grad (Node): the next node gradient in the computation graph

        Returns:
            list: gradient contribute to each input node.
        """
        return [output_grad, output_grad] 
        
class MulOp(Op):
    """ MulOp represents element-wise multiply two nodes. """
    def __call__(self, node_A, node_B):
        """
        Args:
            node_A (Node): first add node.
            node_B (Node): second add node.
        Returns:
            a new node after multiply operation.
        """
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = f"{node_A.name}*{node_B.name}"
        return new_node

    def compute(self, node, input_vals):
        """Given two numbers return the element-wise multiplication of the two numbers.
        Args:
            node (Node): the operation of inputs.
            input_vals (list): the values of inputs.

        Returns:
            the element-wise multiplication of the two numbers.
        """
        assert len(node.inputs) == 2 and len(input_vals) == 2
        return input_vals[0] * input_vals[1]
    
    def gradients(self, node, output_grad):
        """Given gradient of multiply node, return gradient contributions to each input."""
        return [node.inputs[1] * output_grad, node.inputs[0] * output_grad]
    
class AddByConstOp(Op):
    """ AddByConstOp represents add a const to a node. """
    def __call__(self, node_A, const_val):
        """

        Args:
            node_A (Node): the node of add a const.
            const_val: the const number.

        Returns:
            Node: the new node after add by const operation.
        """
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = f"{node_A.name}+{str(const_val)}"
        new_node.const_attr = const_val
        return new_node
    
    def compute(self, node, input_vals):
        assert len(node.inputs) == 1 and len(input_vals) == 1
        return input_vals[0] + node.const_attr
    
    def gradients(self, node, output_grad):
        """Given gradient of add node, return gradient contribution to input."""
        return [output_grad]
    
class MulByConstOp(Op):
    """ MulByConstOp represents multiply with a const operation. """
    def __call__(self, node_A, const_val):
        """
        Args:
            node_A (Node): the node of multiply a const.
            const_val : the const number.

        Returns:
            Node: the new node after multiply by const operation.
        """
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.const_attr = const_val
        new_node.name = f"{node_A.name}*{str(const_val)}"
        return new_node
    
    def compute(self, node, input_vals):
        assert len(node.inputs) == 1 and len(input_vals) == 1
        return input_vals[0] *+ node.const_attr
    
    def gradients(self, node, output_grad):
        """Given gradient of multiplication node, return gradient contribution to input."""
        return [node.const_attr * output_grad]
    
class MatMulOp(Op):
    """ The MatMulOp represents the operation of multiply of two matrix node"""
    def __call__(self, node_A, node_B, transpose_A, transpose_B):
        """
        Args:
            node_A (Node): the first node of the Matmul operation.
            node_B (Node): the second node of the Matmul operation.
            transpose_A (bool): whether to transpose the node A.
            transpose_B (bool): whether to transpose the node B.

        Returns:
            Node: a node that is the result a matrix multiple of two input nodes.
        """
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = f"MatMul({node_A.name},{node_B.name},{str(transpose_A)},{str(transpose_B)})"
        new_node.matmul_attr_trans_A = transpose_A
        new_node.matmul_attr_trans_B = transpose_B
        return new_node
    
    def compute(self, node, input_vals):
        assert len(node.inputs) == 2 and len(input_vals) == 2 
        return np.matmul(input_vals[0], input_vals[1])
    
    def gradients(self, node, output_grad):
        return [np.matmul(output_grad, node.inputs[1].transpose()), np.matmul(node.inputs[0].transpose(), output_grad)]

class ZeroLikeOp(Op):
    """ the ZeroLikeOp represents a constant np.zeros_like. """
    def __call__(self, node_A):
        """Creates a node that represents a np.zeros array of same shape as node_A.

        Args:
            node_A (Node): the input node(you want to generate a node with the same shape with the input node)

        Returns:
            Node: the output node with the same shape with the input node, and it fully contained zero.
        """
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = f"Zeroslike({node_A.name})"
        return new_node
    
    def compute(self, node, input_vals):
        return np.zeros(input_vals[0].shape)
    
    def gradients(self, node, output_grad):
        return [zeros_like_op(node.inputs[0])]
    
class OnesLikeOp(Op):
    """ The OnesLikeOp represents a constant np.ones_like. """
    def __call__(self, node_A):
        """Creates a node that represents a np.ones array of same shape as node_A.

        Args:
            node_A (Node):the input node(you want to generate a node with the same shape with the input node)

        Returns:
            Node: the output node with the same shape with the input node, and it fully contained 1.
        """
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = f"Oneslike({node_A.name})"
        return new_node
    
    def compute(self, node, input_vals):
        return np.ones(input_vals[0].shape)
    
    def gradients(self, node, output_grad):
        return [zeros_like_op(node.inputs[0])]
    
class ExpOp(Op):
    """ The ExpOp represents the operation of expoential """
    def __call__(self, node_A):
        """
        Args:
            node_A (Node): the input node.

        Returns:
            Node: the new node after exp operation.
        """
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = f"exp({node_A.name})"
        return new_node
    
    def compute(self, node, input_vals):
        assert len(node.inputs) == 1 and len(input_vals) == 1
        return np.exp(input_vals[0])

    def gradients(self, node, output_grad):
        """y = exp(x):  dL/dx = (dL/dy)*exp(x) = output_grad * node"""
        return [output_grad * node]
    
class DivOp(Op):
    """ The DivOp represents the operation of division """
    def __call__(self, node_A, node_B):
        """
        Args:
            node_A (Node): the input node 1.
            node_B (Node): the input node 2.
        Returns:
            Node: the new node after node_A / node_B.
        """
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = f"{node_A.name}/{node_B.name}"
        return new_node
    
    def compute(self, node, input_vals):
        assert len(node.inputs) == 2 and len(input_vals) == 2 and (input_vals[1] != 0).all(), "divide by zero"
        return input_vals[0] / input_vals[1]
    
    def gradients(self, node, output_grad):
        return [output_grad / node.inputs[1], - output_grad * node.inputs[0] / (node.inputs[1] * node.inputs[1])]

class NegateOp(Op):
    def __call__(self, node_A):
        new_node =  Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = f"-{node_A.name}"
        return new_node
    
    def compute(self, node, input_vals):
        assert len(node.inputs) == 1 and len(input_vals) == 1
        return - input_vals[0]
    
    def gradients(self, node, output_grad):
        return [-output_grad]

class SubOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = f"{node_A.name}-{node_B.name}"
        return new_node
    
    def compute(self, node, input_vals):
        assert len(node.inputs) == 2 and len(input_vals) == 2
        return input_vals[0] - input_vals[1]
    
    def gradients(self, node, output_grad):
        return [output_grad, -output_grad]

def Variable(name):
    """User defined variables in an expression.  
        e.g. x = Variable(name = "x")
    """
    placeholder_node = placeholder_op()
    placeholder_node.name = name
    return placeholder_node

add_op = AddOp()
mul_op = MulOp()
add_byconst_op = AddByConstOp()
mul_byconst_op = MulByConstOp()
div_op = DivOp()
exp_op = ExpOp()
placeholder_op = PlaceholderOp()
oneslike_op = OnesLikeOp()
zeros_like_op = ZeroLikeOp()
negate_op = NegateOp()
sub_op = SubOp()