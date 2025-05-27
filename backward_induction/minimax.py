class Node:
    """
    A class representing a node in a two-player zero-sum game tree.

    Attributes:
        value (float or int or None): Payoff for Player 1 at terminal nodes.
        children (list of Node or None): Child nodes for non-terminal nodes.
    """

    def __init__(self, value=None, children=None):
        self.value = value
        self.children = children or []


def is_terminal(node):
    """
    Check if a node is a terminal node (no children).
    """
    return not node.children


def minimax(node, maximizing_player):
    """
    Standard minimax algorithm for zero-sum games.

    Args:
        node (Node): Current game tree node.
        maximizing_player (bool): True if it's Player 1's turn; False for Player 2.

    Returns:
        float or int: The minimax value (Player 1's payoff).
    """
    if is_terminal(node):
        return node.value

    if maximizing_player:
        # Player 1 maximizes
        return max(minimax(child, False) for child in node.children)
    else:
        # Player 2 minimizes Player 1's payoff
        return min(minimax(child, True) for child in node.children)


def minimax_with_move(node, maximizing_player):
    """
    Minimax that also returns the best move index for the current player.

    Args:
        node (Node): Current game tree node.
        maximizing_player (bool): True if it's Player 1's turn; False for Player 2.

    Returns:
        tuple: (best_value, best_index)
            best_value: minimax value
            best_index: index in node.children of optimal move (None for terminal)
    """
    if is_terminal(node):
        return node.value, None

    best_index = None
    if maximizing_player:
        best_value = float('-inf')
        for idx, child in enumerate(node.children):
            val, _ = minimax_with_move(child, False)
            if val > best_value:
                best_value = val
                best_index = idx
    else:
        best_value = float('inf')
        for idx, child in enumerate(node.children):
            val, _ = minimax_with_move(child, True)
            if val < best_value:
                best_value = val
                best_index = idx

    return best_value, best_index


# Example: Constructing a game tree
# Structure:
#           (root, P1)
#            /     \
#        (P2)      (P2)
#       /    \    /    \
#    (P1)    5  3      8
#   /   \
#  2     7

# Leaves
leaf2 = Node(value=2)
leaf7 = Node(value=7)
leaf5 = Node(value=5)
leaf3 = Node(value=3)
leaf8 = Node(value=8)

# P1 node under left branch of P2
p1_subnode = Node(children=[leaf2, leaf7])

# P2 nodes
p2_left = Node(children=[p1_subnode, leaf5])
p2_right = Node(children=[leaf3, leaf8])

# Root P1 node
root = Node(children=[p2_left, p2_right])

# Run minimax
value = minimax(root, True)
value_with_move, move_index = minimax_with_move(root, True)

print(f"Minimax value (Player 1 payoff): {value}")
print(f"Minimax with move: value={value_with_move}, best move index={move_index}")

