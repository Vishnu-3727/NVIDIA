"""
RRT (Rapidly-exploring Random Tree) Algorithm Implementation
Standard RRT algorithm without rewiring optimization.
"""

import numpy as np
from typing import List, Tuple, Optional
from map import Map
from utils import Node, euclidean_distance, random_sample, nearest_node, steer, extract_path


class RRT:
    """
    RRT (Rapidly-exploring Random Tree) path planner.
    
    Attributes:
        start (Tuple[float, float]): Start position (x, y)
        goal (Tuple[float, float]): Goal position (x, y)
        map (Map): Environment map
        max_iterations (int): Maximum iterations
        step_size (float): Maximum distance to steer in each iteration
        goal_threshold (float): Distance threshold to consider goal reached
        goal_bias (float): Probability of sampling goal
        nodes (List[Node]): All nodes in the tree
        goal_node (Optional[Node]): Goal node when found
    """
    
    def __init__(self, start: Tuple[float, float], goal: Tuple[float, float], 
                 environment_map: Map, max_iterations: int = 5000,
                 step_size: float = 2.0, goal_threshold: float = 1.5,
                 goal_bias: float = 0.1):
        """
        Initialize RRT planner.
        
        Args:
            start: Start position (x, y)
            goal: Goal position (x, y)
            environment_map: Map object
            max_iterations: Maximum iterations
            step_size: Maximum steering distance
            goal_threshold: Distance to goal to consider it reached
            goal_bias: Probability of sampling goal
        """
        self.start = start
        self.goal = goal
        self.map = environment_map
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_threshold = goal_threshold
        self.goal_bias = goal_bias
        
        # Initialize tree with start node
        self.nodes: List[Node] = [Node(start[0], start[1], parent=None, cost=0.0)]
        self.goal_node: Optional[Node] = None
    
    def plan(self) -> bool:
        """
        Execute RRT planning algorithm.
        
        Returns:
            True if path found, False otherwise
        """
        for iteration in range(self.max_iterations):
            # Sample random point with goal bias
            random_point = random_sample(self.map.width, self.map.height, 
                                        self.goal, self.goal_bias)
            
            # Find nearest node in tree
            nearest = nearest_node(self.nodes, random_point)
            
            # Steer toward random point
            new_point = steer(nearest, random_point, self.step_size)
            
            # Check collision
            if self.map.is_collision_free(nearest.x, nearest.y, 
                                         new_point[0], new_point[1]):
                # Create new node
                new_node = Node(
                    new_point[0], new_point[1],
                    parent=nearest,
                    cost=nearest.cost + euclidean_distance(nearest.x, nearest.y,
                                                          new_point[0], new_point[1])
                )
                self.nodes.append(new_node)
                
                # Check if goal is reached
                distance_to_goal = euclidean_distance(new_point[0], new_point[1],
                                                     self.goal[0], self.goal[1])
                if distance_to_goal < self.goal_threshold:
                    # Connect to goal
                    goal_node = Node(
                        self.goal[0], self.goal[1],
                        parent=new_node,
                        cost=new_node.cost + distance_to_goal
                    )
                    self.nodes.append(goal_node)
                    self.goal_node = goal_node
                    return True
            
            # Log progress every 500 iterations
            if (iteration + 1) % 500 == 0:
                print(f"  RRT: {iteration + 1}/{self.max_iterations} iterations, "
                      f"{len(self.nodes)} nodes")
        
        return False
    
    def get_path(self) -> Optional[List[Tuple[float, float]]]:
        """
        Extract the path from start to goal.
        
        Returns:
            Path as list of (x, y) tuples, or None if no path found
        """
        if self.goal_node is None:
            return None
        return extract_path(self.goal_node)
    
    def get_path_length(self) -> Optional[float]:
        """
        Calculate total path length.
        
        Returns:
            Path length or None if no path found
        """
        if self.goal_node is None:
            return None
        return self.goal_node.cost
    
    def get_nodes(self) -> List[Node]:
        """
        Get all nodes in the tree.
        
        Returns:
            List of all nodes
        """
        return self.nodes.copy()
    
    def get_tree_edges(self) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        Get all edges in the tree for visualization.
        
        Returns:
            List of edges as ((x1, y1), (x2, y2)) tuples
        """
        edges = []
        for node in self.nodes:
            if node.parent is not None:
                edges.append(((node.parent.x, node.parent.y), (node.x, node.y)))
        return edges
