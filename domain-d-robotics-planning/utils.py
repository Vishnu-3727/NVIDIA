"""
Utility Functions Module
Contains utility functions for distance calculations, sampling, and path operations.
"""

import numpy as np
from typing import Tuple, List


class Node:
    """
    Represents a node in the RRT/RRT* tree.
    
    Attributes:
        x (float): X-coordinate
        y (float): Y-coordinate
        parent (Node): Parent node in the tree
        cost (float): Cost from root to this node (for RRT*)
    """
    
    def __init__(self, x: float, y: float, parent=None, cost: float = 0.0):
        """
        Initialize a node.
        
        Args:
            x: X-coordinate
            y: Y-coordinate
            parent: Parent node (None for root)
            cost: Cost from root (used in RRT*)
        """
        self.x = x
        self.y = y
        self.parent = parent
        self.cost = cost
    
    def __repr__(self) -> str:
        """String representation of node."""
        return f"Node({self.x:.2f}, {self.y:.2f}, cost={self.cost:.2f})"


def euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """
    Calculate Euclidean distance between two points.
    
    Args:
        x1, y1: First point coordinates
        x2, y2: Second point coordinates
        
    Returns:
        Euclidean distance
    """
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def random_sample(map_width: float, map_height: float, goal: Tuple[float, float],
                 goal_bias: float = 0.1) -> Tuple[float, float]:
    """
    Sample a random point with goal bias.
    With probability goal_bias, return the goal; otherwise random point.
    
    Args:
        map_width: Map width
        map_height: Map height
        goal: Goal position (x, y)
        goal_bias: Probability of sampling goal (0.0 to 1.0)
        
    Returns:
        Random sampled point (x, y)
    """
    if np.random.random() < goal_bias:
        return goal
    else:
        x = np.random.uniform(0, map_width)
        y = np.random.uniform(0, map_height)
        return (x, y)


def nearest_node(node_list: List[Node], random_point: Tuple[float, float]) -> Node:
    """
    Find the nearest node in the tree to a random point.
    
    Args:
        node_list: List of all nodes in the tree
        random_point: Random point (x, y)
        
    Returns:
        The nearest node
    """
    distances = [euclidean_distance(node.x, node.y, random_point[0], random_point[1]) 
                for node in node_list]
    return node_list[np.argmin(distances)]


def steer(from_node: Node, to_point: Tuple[float, float], 
         step_size: float) -> Tuple[float, float]:
    """
    Steer from a node toward a point with maximum step size.
    
    Args:
        from_node: Starting node
        to_point: Target point (x, y)
        step_size: Maximum step size
        
    Returns:
        New point (x, y) after steering
    """
    distance = euclidean_distance(from_node.x, from_node.y, to_point[0], to_point[1])
    
    if distance < step_size:
        return to_point
    else:
        ratio = step_size / distance
        x = from_node.x + ratio * (to_point[0] - from_node.x)
        y = from_node.y + ratio * (to_point[1] - from_node.y)
        return (x, y)


def extract_path(node: Node) -> List[Tuple[float, float]]:
    """
    Extract path from root to given node by backtracking through parents.
    
    Args:
        node: End node
        
    Returns:
        List of points (x, y) from root to node
    """
    path = []
    current = node
    while current is not None:
        path.append((current.x, current.y))
        current = current.parent
    return path[::-1]  # Reverse to get start -> end


def get_neighbors(node_list: List[Node], node: Node, radius: float) -> List[Node]:
    """
    Get all nodes within a certain radius of a given node.
    Used in RRT* for rewiring.
    
    Args:
        node_list: List of all nodes in the tree
        node: Reference node
        radius: Search radius
        
    Returns:
        List of neighbor nodes
    """
    neighbors = []
    for other_node in node_list:
        distance = euclidean_distance(node.x, node.y, other_node.x, other_node.y)
        if distance <= radius:
            neighbors.append(other_node)
    return neighbors


def calculate_search_radius(node_count: int, max_radius: float = 50.0) -> float:
    """
    Calculate search radius for RRT* rewiring based on node count.
    Radius decreases as tree grows.
    
    Args:
        node_count: Current number of nodes in tree
        max_radius: Maximum radius
        
    Returns:
        Search radius for neighbor finding
    """
    # Formula from RRT* paper: r = min(max_radius, C * (log(n) / n)^(1/d))
    # For 2D: d=2
    if node_count < 2:
        return max_radius
    
    dimension = 2
    return min(max_radius, 3.0 * np.sqrt(np.log(node_count) / node_count))


def smooth_path(path: List[Tuple[float, float]], environment_map) -> List[Tuple[float, float]]:
    """
    Post-process path to remove unnecessary waypoints and smooth it.
    
    Args:
        path: Original path as list of (x, y) tuples
        environment_map: Map object for collision checking
        
    Returns:
        Smoothed path with fewer waypoints
    """
    if len(path) <= 2:
        return path
    
    smoothed = [path[0]]
    current_idx = 0
    
    while current_idx < len(path) - 1:
        # Try to connect to furthest point that's collision-free
        furthest_idx = len(path) - 1
        found = False
        
        for next_idx in range(len(path) - 1, current_idx + 1, -1):
            if environment_map.is_collision_free(
                path[current_idx][0], path[current_idx][1],
                path[next_idx][0], path[next_idx][1]
            ):
                smoothed.append(path[next_idx])
                current_idx = next_idx
                found = True
                break
        
        if not found:
            current_idx += 1
            if current_idx < len(path):
                smoothed.append(path[current_idx])
    
    return smoothed
