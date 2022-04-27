def find_topo_order(eval_node_list):
    """
    because the comutation graph not have cycle.so the Post-order DFS can find the topo order.
    this is implementation of the post-order DFS.

    Args:
        eval_node_list (list): the nodes we need to compute.

    Returns:
        list: the order of the node.
    """
    visited = set()
    topo_order = []
    for eval_node in eval_node_list:
        topo_sort_dfs(eval_node, visited, topo_order)
    return topo_order


def topo_sort_dfs(node, visited, topo_order):
    if node in visited:
        return 
    visited.add(node)
    for n in node.inputs:
        topo_sort_dfs(n, visited, topo_order)
    topo_order.append(node)