"""
A generic implementation of a branch-and-bound.
"""
import abc
from abc import abstractmethod
import random
import heapq

class BranchingRule(abc):
    """
    An abstract branching rule.
    """
    def __init__(self):
        """
        The branching rule's initializer
        """
        pass

    @abc.abstractmethod
    def get_branching_variable(self):
        """
        Selects and returns an object to branch on.
        """
        pass

class RandomBranchingRule(BranchingRule):
    """
    A branching rule that picks a variable to branch on randomly.
    """
    def __init__(self):
        """
        The branching rule's initializer
        """
        pass

    def get_branching_variable(self, candidates: list):
        """
        Selects and returns an object to branch on randomly from a list of candidates.
        """
        assert candidates, "Cannot randomly draw from an empty list of candidates."
        return random.choice(candidates)
        

class Node(abc):
    """
    A node of the branch and bound.
    """

    def __init__(self, local_lower_bound: float):
        """
        Initialises a node of the branch and bound with a lower bound.
        """
        self.local_lower_bound = local_lower_bound

    @abc.abstractmethod
    def create_children(self, branched_object: object):
        """
        Creates and returns two children nodes that partition
        the space of solutions according to the given branching object.
        """
        pass

    @abc.abstractmethod
    def is_feasible(self):
        """
        Checks if the node represents a feasible solution.
        """
        pass

    @abc.abstractmethod
    def compute_upper_bound(self):
        """
        Computes an upper bound for the node.
        """
        pass

class BranchAndBound:
    """
    An implementation of the Branch-and-Bound
    """
    def __init__(self, branching_rule: BranchingRule):
        """
        Initializes the B&B with a branching rule.
        """
        self.branching_rule = branching_rule
        self.best_upper_bound = float("inf")
        self.best_solution = None
        self.priority_queue = []  # Min-heap for nodes sorted by lower bound

    def solve(self, root: Node):
        """
        Solves the optimization problem using branch-and-bound.
        """
        heapq.heappush(self.priority_queue, (root.local_lower_bound, root))
        
        while self.priority_queue:
            _, node = heapq.heappop(self.priority_queue)
            
            # Prune if the node's bound is worse than the best found
            if node.local_lower_bound >= self.best_upper_bound:
                continue
            
            # If feasible, check if it's the best solution found
            if node.is_feasible():
                upper_bound = node.compute_upper_bound()
                if upper_bound < self.best_upper_bound:
                    self.best_upper_bound = upper_bound
                    self.best_solution = node
                continue
            
            # Branch if not feasible
            candidates = self.get_branching_candidates(node)
            if not candidates:
                continue
            
            branching_object = self.branching_rule.get_branching_variable(candidates)
            children = node.create_children(branching_object)
            
            for child in children:
                if child.local_lower_bound < self.best_upper_bound:
                    heapq.heappush(self.priority_queue, (child.local_lower_bound, child))
        
        return self.best_solution, self.best_upper_bound
    
    @abc.abstractmethod
    def get_branching_candidates(self, node: Node):
        """
        Returns the candidates for branching from a node.
        """
        pass
