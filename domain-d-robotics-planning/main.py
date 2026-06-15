"""
Main Execution File
Complete path planning pipeline: Input → Planning → Extraction → Visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import List, Tuple, Optional
from map import Map
from rrt import RRT
from rrt_star import RRTStar
from utils import smooth_path


class PathPlanningVisualizer:
    """
    Visualizes path planning results with matplotlib.
    """
    
    def __init__(self, map_obj: Map, figsize: Tuple[int, int] = (15, 5)):
        """
        Initialize visualizer.
        
        Args:
            map_obj: Map object
            figsize: Figure size (width, height)
        """
        self.map = map_obj
        self.figsize = figsize
    
    def visualize_single(self, planner, title: str, smooth: bool = False) -> None:
        """
        Visualize a single planner result.
        
        Args:
            planner: RRT or RRT* planner object
            title: Plot title
            smooth: Whether to smooth the path
        """
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        # Draw obstacles
        self._draw_obstacles(ax)
        
        # Draw tree edges
        for edge in planner.get_tree_edges():
            ax.plot([edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]],
                   'b-', linewidth=0.5, alpha=0.3)
        
        # Draw all nodes
        nodes = planner.get_nodes()
        node_xs = [node.x for node in nodes]
        node_ys = [node.y for node in nodes]
        ax.scatter(node_xs, node_ys, c='blue', s=10, alpha=0.5)
        
        # Draw start and goal
        ax.plot(self.map.width * 0.1, self.map.height * 0.1, 'g*', markersize=20,
               label='Start')
        ax.plot(self.map.width * 0.9, self.map.height * 0.9, 'r*', markersize=20,
               label='Goal')
        
        # Draw path if found
        path = planner.get_path()
        if path:
            if smooth:
                path = smooth_path(path, self.map)
            
            path_xs = [p[0] for p in path]
            path_ys = [p[1] for p in path]
            ax.plot(path_xs, path_ys, 'r-', linewidth=3, label='Path',
                   alpha=0.8)
            
            # Mark waypoints
            ax.scatter(path_xs, path_ys, c='red', s=50, marker='o', zorder=5)
        
        ax.set_xlim(0, self.map.width)
        ax.set_ylim(0, self.map.height)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(title)
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def visualize_comparison(self, rrt_planner, rrt_star_planner,
                            smooth: bool = False) -> None:
        """
        Visualize RRT and RRT* side by side.
        
        Args:
            rrt_planner: RRT planner object
            rrt_star_planner: RRT* planner object
            smooth: Whether to smooth paths
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # RRT visualization
        self._draw_single_plan(ax1, rrt_planner, "RRT", smooth)
        
        # RRT* visualization
        self._draw_single_plan(ax2, rrt_star_planner, "RRT*", smooth)
        
        plt.tight_layout()
        return fig
    
    def _draw_single_plan(self, ax, planner, title: str, smooth: bool = False) -> None:
        """
        Draw a single planner result on axis.
        
        Args:
            ax: Matplotlib axis
            planner: RRT or RRT* planner
            title: Plot title
            smooth: Whether to smooth the path
        """
        # Draw obstacles
        for obs_x, obs_y, obs_w, obs_h in self.map.obstacles:
            rect = Rectangle((obs_x - obs_w/2, obs_y - obs_h/2), obs_w, obs_h,
                           linewidth=2, edgecolor='black', facecolor='gray',
                           alpha=0.7)
            ax.add_patch(rect)
        
        # Draw tree edges
        for edge in planner.get_tree_edges():
            ax.plot([edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]],
                   'b-', linewidth=0.5, alpha=0.2)
        
        # Draw all nodes
        nodes = planner.get_nodes()
        node_xs = [node.x for node in nodes]
        node_ys = [node.y for node in nodes]
        ax.scatter(node_xs, node_ys, c='lightblue', s=5, alpha=0.4)
        
        # Draw start and goal
        ax.plot(0.1 * self.map.width, 0.1 * self.map.height, 'g*',
               markersize=25, label='Start', zorder=5)
        ax.plot(0.9 * self.map.width, 0.9 * self.map.height, 'r*',
               markersize=25, label='Goal', zorder=5)
        
        # Draw path if found
        path = planner.get_path()
        if path:
            if smooth:
                path = smooth_path(path, self.map)
            
            path_xs = [p[0] for p in path]
            path_ys = [p[1] for p in path]
            ax.plot(path_xs, path_ys, 'r-', linewidth=2.5, label='Path',
                   alpha=0.9, zorder=4)
            
            # Mark waypoints
            ax.scatter(path_xs, path_ys, c='red', s=30, marker='o',
                      zorder=6, edgecolors='darkred', linewidth=1)
            
            path_length = planner.get_path_length()
            ax.text(0.5, 0.95, f"Path Length: {path_length:.2f}",
                   transform=ax.transAxes, ha='center',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        else:
            ax.text(0.5, 0.95, "No path found",
                   transform=ax.transAxes, ha='center',
                   bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        
        ax.set_xlim(0, self.map.width)
        ax.set_ylim(0, self.map.height)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        ax.set_xlabel('X', fontsize=10)
        ax.set_ylabel('Y', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
    
    def _draw_obstacles(self, ax) -> None:
        """Draw all obstacles on axis."""
        for obs_x, obs_y, obs_w, obs_h in self.map.obstacles:
            rect = Rectangle((obs_x - obs_w/2, obs_y - obs_h/2), obs_w, obs_h,
                           linewidth=2, edgecolor='black', facecolor='gray',
                           alpha=0.7)
            ax.add_patch(rect)


def run_path_planning_pipeline():
    """
    Complete path planning pipeline.
    Input → Planning → Extraction → Visualization
    """
    print("=" * 60)
    print("ROBOTICS PATH PLANNING SYSTEM - RRT / RRT*")
    print("=" * 60)
    
    # ===== INPUT: Create environment =====
    print("\n[1] INITIALIZATION")
    print("-" * 60)
    
    # Create map
    map_width = 100.0
    map_height = 100.0
    environment_map = Map(map_width, map_height)
    
    # Add obstacles
    print(f"  Map size: {map_width} x {map_height}")
    print("  Adding obstacles...")
    
    # Manually defined obstacles for deterministic test
    obstacles = [
        (30, 30, 15, 15),
        (70, 70, 15, 15),
        (50, 50, 12, 12),
        (20, 70, 10, 25),
        (80, 25, 15, 20),
        (40, 80, 12, 12),
    ]
    
    for obs in obstacles:
        environment_map.add_obstacle(*obs)
    
    print(f"  Total obstacles: {len(environment_map.obstacles)}")
    
    # Define start and goal
    start = (10.0, 10.0)
    goal = (90.0, 90.0)
    print(f"  Start: {start}")
    print(f"  Goal: {goal}")
    
    # ===== PLANNING: Run RRT and RRT* =====
    print("\n[2] PATH PLANNING")
    print("-" * 60)
    
    # RRT planning
    print("\n  Running RRT...")
    rrt_planner = RRT(
        start=start,
        goal=goal,
        environment_map=environment_map,
        max_iterations=5000,
        step_size=2.0,
        goal_threshold=1.5,
        goal_bias=0.15
    )
    
    rrt_found = rrt_planner.plan()
    rrt_nodes = len(rrt_planner.get_nodes())
    rrt_path_length = rrt_planner.get_path_length()
    
    print(f"  ✓ RRT completed")
    print(f"    Nodes created: {rrt_nodes}")
    print(f"    Path found: {rrt_found}")
    if rrt_found:
        print(f"    Path length: {rrt_path_length:.2f}")
    
    # RRT* planning
    print("\n  Running RRT*...")
    rrt_star_planner = RRTStar(
        start=start,
        goal=goal,
        environment_map=environment_map,
        max_iterations=5000,
        step_size=2.0,
        goal_threshold=1.5,
        goal_bias=0.15,
        max_radius=50.0
    )
    
    rrt_star_found = rrt_star_planner.plan()
    rrt_star_nodes = len(rrt_star_planner.get_nodes())
    rrt_star_path_length = rrt_star_planner.get_path_length()
    
    print(f"  ✓ RRT* completed")
    print(f"    Nodes created: {rrt_star_nodes}")
    print(f"    Path found: {rrt_star_found}")
    if rrt_star_found:
        print(f"    Path length: {rrt_star_path_length:.2f}")
    
    # ===== EXTRACTION: Process results =====
    print("\n[3] PATH EXTRACTION & PROCESSING")
    print("-" * 60)
    
    if rrt_found:
        rrt_path = rrt_planner.get_path()
        rrt_path_smooth = smooth_path(rrt_path, environment_map)
        print(f"  RRT:")
        print(f"    Original waypoints: {len(rrt_path)}")
        print(f"    Smoothed waypoints: {len(rrt_path_smooth)}")
    
    if rrt_star_found:
        rrt_star_path = rrt_star_planner.get_path()
        rrt_star_path_smooth = smooth_path(rrt_star_path, environment_map)
        print(f"  RRT*:")
        print(f"    Original waypoints: {len(rrt_star_path)}")
        print(f"    Smoothed waypoints: {len(rrt_star_path_smooth)}")
    
    # ===== VISUALIZATION =====
    print("\n[4] VISUALIZATION")
    print("-" * 60)
    
    visualizer = PathPlanningVisualizer(environment_map)
    
    # Comparison plot
    print("  Creating comparison visualization...")
    fig = visualizer.visualize_comparison(rrt_planner, rrt_star_planner, smooth=False)
    plt.savefig('path_planning_comparison.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: path_planning_comparison.png")
    
    # Summary statistics
    print("\n[5] SUMMARY STATISTICS")
    print("-" * 60)
    
    if rrt_found and rrt_star_found:
        improvement = ((rrt_path_length - rrt_star_path_length) / rrt_path_length) * 100
        print(f"  RRT path length: {rrt_path_length:.2f}")
        print(f"  RRT* path length: {rrt_star_path_length:.2f}")
        print(f"  Improvement: {improvement:.1f}%")
    
    print("\n" + "=" * 60)
    print("Pipeline execution completed successfully!")
    print("=" * 60)
    
    plt.show()


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run complete pipeline
    run_path_planning_pipeline()
