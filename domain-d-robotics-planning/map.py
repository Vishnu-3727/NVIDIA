"""
Map Environment Module
Handles 2D grid-based environment, obstacles, and collision detection.
"""

import numpy as np
from typing import List, Tuple


class Map:
    """
    Represents a 2D rectangular map with obstacles.
    
    Attributes:
        width (float): Map width
        height (float): Map height
        obstacles (List[Tuple]): List of rectangular obstacles (x, y, width, height)
    """
    
    def __init__(self, width: float = 100.0, height: float = 100.0):
        """
        Initialize the map.
        
        Args:
            width: Map width in units
            height: Map height in units
        """
        self.width = width
        self.height = height
        self.obstacles: List[Tuple[float, float, float, float]] = []
    
    def add_obstacle(self, x: float, y: float, width: float, height: float) -> None:
        """
        Add a rectangular obstacle to the map.
        
        Args:
            x: X-coordinate of obstacle center
            y: Y-coordinate of obstacle center
            width: Width of obstacle
            height: Height of obstacle
        """
        self.obstacles.append((x, y, width, height))
    
    def add_random_obstacles(self, num_obstacles: int = 10, 
                            min_size: float = 2.0, 
                            max_size: float = 8.0) -> None:
        """
        Add random obstacles to the map.
        
        Args:
            num_obstacles: Number of obstacles to add
            min_size: Minimum obstacle size
            max_size: Maximum obstacle size
        """
        for _ in range(num_obstacles):
            x = np.random.uniform(min_size, self.width - min_size)
            y = np.random.uniform(min_size, self.height - min_size)
            size = np.random.uniform(min_size, max_size)
            self.add_obstacle(x, y, size, size)
    
    def is_in_bounds(self, x: float, y: float) -> bool:
        """
        Check if a point is within map boundaries.
        
        Args:
            x: X-coordinate
            y: Y-coordinate
            
        Returns:
            True if point is within bounds, False otherwise
        """
        return 0 <= x <= self.width and 0 <= y <= self.height
    
    def is_collision_free(self, x1: float, y1: float, x2: float, y2: float, 
                         step_size: float = 0.5) -> bool:
        """
        Check if a line segment from (x1, y1) to (x2, y2) is collision-free.
        Uses discrete sampling along the line segment.
        
        Args:
            x1: Start x-coordinate
            y1: Start y-coordinate
            x2: End x-coordinate
            y2: End y-coordinate
            step_size: Size of steps for collision checking
            
        Returns:
            True if path is collision-free, False otherwise
        """
        # Calculate distance and number of steps
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        num_steps = max(int(distance / step_size), 1)
        
        # Check each step along the path
        for i in range(num_steps + 1):
            t = i / num_steps if num_steps > 0 else 0
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            
            # Check bounds
            if not self.is_in_bounds(x, y):
                return False
            
            # Check collision with obstacles
            if self._is_point_in_obstacle(x, y):
                return False
        
        return True
    
    def _is_point_in_obstacle(self, x: float, y: float) -> bool:
        """
        Check if a point collides with any obstacle.
        
        Args:
            x: X-coordinate
            y: Y-coordinate
            
        Returns:
            True if point is in obstacle, False otherwise
        """
        for obs_x, obs_y, obs_w, obs_h in self.obstacles:
            # Rectangle collision check with padding for robot radius
            if (obs_x - obs_w/2 <= x <= obs_x + obs_w/2 and
                obs_y - obs_h/2 <= y <= obs_y + obs_h/2):
                return True
        return False
    
    def get_obstacles(self) -> List[Tuple[float, float, float, float]]:
        """
        Get list of all obstacles.
        
        Returns:
            List of obstacles as (x, y, width, height) tuples
        """
        return self.obstacles.copy()
