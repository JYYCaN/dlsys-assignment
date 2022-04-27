from utils import find_topo_order
import autodiff as ad
class Executor:
    """"""
    def __init__(self, eval_node_list):
        """
        Args:
            eval_node_list (Node): list of nodes whose values need to be computed.
        """
        self.eval_node_list = eval_node_list
        
    def run(self, feed_dict: dict):
        
        node_to_val_map = dict(feed_dict)
        
        # get the topo order of computation graph nodes
        topo_order = find_topo_order(self.eval_node_list)
        
        for node in topo_order:
            if node not in node_to_val_map.keys():
                input_vals = [node_to_val_map[n] for n in node.inputs]
                
                value = node.op.compute(node, input_vals)
                node_to_val_map[node] = value
        eval_node_val_resluts = [node_to_val_map[node] for node in self.eval_node_list]
        return eval_node_val_resluts
    
def gradients(output_node, node_list):
    # a map from node to a list of output_node from each output node
    node_to_output_grad_list = {}
    node_to_output_grad_list[output_node] = [ad.oneslike_op(output_node)]
    # a map from node to the gradient of that node
    # node_to_output_grad = {}
    reverse_topo_order = reversed(find_topo_order([output_node]))
    for node in reverse_topo_order:
        for i, input in enumerate(node.inputs):
            if input not in node_to_output_grad_list.keys():
                node_to_output_grad_list[input] = [node.op.gradients(node, sum_node_list(node_to_output_grad_list[node]))[i]]
            else:
                node_to_output_grad_list[input].append(node.op.gradients(node, sum_node_list(node_to_output_grad_list[node]))[i])
                
    reslut_node_list = [sum_node_list(node_to_output_grad_list[node]) for node in node_list]
    return reslut_node_list

def sum_node_list(node_list):
    """Custom sum function in order to avoid create redundant nodes in Python sum implementation."""
    from operator import add
    from functools import reduce
    return reduce(add, node_list)