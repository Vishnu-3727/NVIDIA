"""
RRT* (RRT-Star) Algorithm Implementation
RRT with asymptotically optimal rewiring.
"""

import numpy as np
from typing import List, Tuple, Optional
from map import Map
from utils import (Node, euclidean_distance, random_sample, nearest_node, steer,
                   extract_path, get_neighbors, calculate_search_radius)


class RRTStar:
    """
    RRT* (RRT-Star) path planner with rewiring optimization.
    
    Attributes:
        start (Tuple[float, float]): Start position (x, y)
        goal (Tuple[float, float]): Goal position (x, y)
        map (Map): Environment map
        max_iterations (int): Maximum iterations
        step_size (float): Maximum distance to steer in each iteration
        goal_threshold (float): Distance threshold to consider goal reached
        goal_bias (float): Probability of sampling goal
        max_radius (float): Maximum radius for rewiring search
        nodes (List[Node]): All nodes in the tree
        goal_node (Optional[Node]): Best goal node found
    """
    
    def __init__(self, start: Tuple[float, float], goal: Tuple[float, float],
                 environment_map: Map, max_iterations: int = 5000,
                 step_size: float = 2.0, goal_threshold: float = 1.5,
                 goal_bias: float = 0.1, max_radius: float = 50.0):
        """
        Initialize RRT* planner.
        
        Args:
            start: Start position (x, y)
            goal: Goal position (x, y)
            environment_map: Map object
            max_iterations: Maximum iterations
            step_size: Maximum steering distance
            goal_threshold: Distance to goal to consider it reached
            goal_bias: Probability of sampling goal
            max_radius: Maximum search radius for rewiring
        """
        self.start = start
        self.goal = goal
        self.map = environment_map
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_threshold = goal_threshold
        self.goal_bias = goal_bias
        self.max_radius = max_radius
        
        # Initialize tree with start node
        self.nodes: List[Node] = [Node(start[0], start[1], parent=None, cost=0.0)]
        self.goal_node: Optional[Node] = None
    
    def plan(self) -> bool:
        """
        Execute RRT* planning algorithm with rewiring.
        
        Returns:
            True if path found, False otherwise
        """
        path_found = False
        
        for iteration in range(self.max_iterations):
            # Sample random point with goal bias
            random_point = random_sample(self.map.width, self.map.height,
                                        self.goal, self.goal_bias)
            
            # Find nearest node in tree
            nearest = nearest_node(self.nodes, random_point)
            
            # Steer toward random point
            new_point = steer(nearest, random_point, self.step_size)
            
            # Check collision with nearest node
            if self.map.is_collision_free(nearest.x, nearest.y,
                                         new_point[0], new_point[1]):
                
                # Calculate cost from nearest node
                cost_from_nearest = (nearest.cost + 
                                   euclidean_distance(nearest.x, nearest.y,
                                                    new_point[0], new_point[1]))
                
                # Create new node
                new_node = Node(new_point[0], new_point[1],
                              parent=nearest,
                              cost=cost_from_nearest)
                
                # Find neighbors within search radius
                search_radius = calculate_search_radius(len(self.nodes), self.max_radius)
                neighbors = get_neighbors(self.nodes, new_node, search_radius)
                
                # Find best parent among neighbors
                best_parent = nearest
                best_cost = cost_from_nearest
                
                for neighbor in neighbors:
                    # Skip if neighbor is the nearest node (already considered)
                    if neighbor == nearest:
                        continue
                    
                    # Check if we can connect to this neighbor
                    if self.map.is_collision_free(neighbor.x, neighbor.y,
                                                 new_point[0], new_point[1]):
                        # Calculate cost through this neighbor
                        distance_to_neighbor = euclidean_distance(neighbor.x, neighbor.y,
                                                                new_point[0], new_point[1])
                        cost_through_neighbor = neighbor.cost + distance_to_neighbor
                        
                        # If this path is better, update parent
                        if cost_through_neighbor < best_cost:
                            best_cost = cost_through_neighbor
                            best_parent = neighbor
                
                # Set best parent
                new_node.parent = best_parent
                new_node.cost = best_cost
                
                self.nodes.append(new_node)
                
                # Rewire: check if new node can improve neighbors
                self._rewire(new_node, neighbors)
                
                # Check if goal is reached
                distance_to_goal = euclidean_distance(new_point[0], new_point[1],
                                                     self.goal[0], self.goal[1])
                if distance_to_goal < self.goal_threshold:
                    # Try to connect to goal
                    if self.map.is_collision_free(new_point[0], new_point[1],
                                                 self.goal[0], self.goal[1]):
                        cost_to_goal = new_node.cost + distance_to_goal
                        
                        # Update goal node if this is better
                        if self.goal_node is None or cost_to_goal < self.goal_node.cost:
                            goal_node = Node(self.goal[0], self.goal[1],
                                           parent=new_node,
                                           cost=cost_to_goal)
                            self.nodes.append(goal_node)
                            self.goal_node = goal_node
                            path_found = True
            
            # Log progress every 500 iterations
            if (iteration + 1) % 500 == 0:
                best_cost = self.goal_node.cost if self.goal_node else float('inf')
                print(f"  RRT*: {iteration + 1}/{self.max_iterations} iterations, "
                      f"{len(self.nodes)} nodes, best cost: {best_cost:.2f}")
        
        return path_found
    
    def _rewire(self, new_node: Node, neighbors: List[Node]) -> None:
        """
        Rewire neighbors if new node provides a better path.
        
        Args:
            new_node: Newly added node
            neighbors: Neighboring nodes within search radius
        """
        for neighbor in neighbors:
            if neighbor == new_node.parent:  # Skip parent
                continue
            
            # Check if we can connect from new node to neighbor
            if self.map.is_collision_free(new_node.x, new_node.y,
                                         neighbor.x, neighbor.y):
                # Calculate cost through new node
                distance_to_neighbor = euclidean_distance(new_node.x, new_node.y,
                                                        neighbor.x, neighbor.y)
                cost_through_new = new_node.cost + distance_to_neighbor
                
                # If this path is better, rewire
                if cost_through_new < neighbor.cost:
                    neighbor.parent = new_node
                    neighbor.cost = cost_through_new
                    
                    # Update costs of all descendants
                    self._update_descendant_costs(neighbor)
    
    def _update_descendant_costs(self, node: Node) -> None:
        """
        Recursively update costs of all descendants after rewiring.
        
        Args:
            node: Node whose descendants to update
        """
        for other_node in self.nodes:
            if other_node.parent == node:
                distance = euclidean_distance(node.x, node.y,
                                            other_node.x, other_node.y)
                other_node.cost = node.cost + distance
                self._update_descendant_costs(other_node)
    
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
