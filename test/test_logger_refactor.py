#!/usr/bin/env python3
"""
Simple test to verify the refactored logger still creates correct parent-child relationships.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.logger import TrajectoryLogger

def test_logger_refactor():
    """Test that the refactored logger automatically creates correct parent-child relationships."""
    logger = TrajectoryLogger()
    
    # Add root node
    node1_id = logger.add_node(phase="start", turn=0, tags={"test": "root"})
    print(f"Root node: {node1_id}")
    print(f"Root ID: {logger.root_id}")
    assert logger.root_id == node1_id
    assert logger.nodes[node1_id].parent_id is None
    
    # Add second node (should be child of first)
    node2_id = logger.add_node(phase="plan/create", turn=1, tags={"test": "child1"})
    print(f"\nSecond node: {node2_id}")
    print(f"Last node ID: {node1_id}")
    assert logger.current_node().id == node2_id
    assert logger.nodes[node2_id].parent_id == node1_id
    assert node2_id in logger.children.get(node1_id, [])
    
    # Add third node (should be child of second)
    node3_id = logger.add_node(phase="act/run", turn=1, tags={"test": "child2"})
    print(f"\nThird node: {node3_id}")
    print(f"Last node ID: {node2_id}")
    assert logger.current_node().id == node3_id
    assert logger.nodes[node3_id].parent_id == node2_id
    assert node3_id in logger.children.get(node2_id, [])
    
    # Verify tree structure
    print("\nTree structure:")
    print(f"Root: {logger.root_id}")
    print(f"Children of {node1_id}: {logger.children.get(node1_id, [])}")
    print(f"Children of {node2_id}: {logger.children.get(node2_id, [])}")
    print(f"Children of {node3_id}: {logger.children.get(node3_id, [])}")
    
    # Verify the chain: root -> child1 -> child2
    assert len(logger.children.get(node1_id, [])) == 1
    assert len(logger.children.get(node2_id, [])) == 1
    assert len(logger.children.get(node3_id, [])) == 0
    
    print("\n✅ All tests passed! The refactored logger works correctly.")

if __name__ == "__main__":
    test_logger_refactor()