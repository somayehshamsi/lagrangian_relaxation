"""
A generic implementation of a branch-and-bound.
"""
import abc
from abc import abstractmethod
import random

class BranchingRule(abc):
    """
    An abstract branching rule.
    """
    def __init__(self):
        """
        The branching rule's initializer
        """
        pass

    @abstractmethod
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

    @abstractmethod
    def create_children(self, branched_object: object):
        """
        Creates and returns two children nodes that partition
        the space of solutions according to the given branching object.
        """
        pass


class BranchAndBound:
    """
    An implementation of the Branch-and-Bound
    """
    def __init__(self, branching_rule: BranchingRule):
        """
        Initialises the B&B with a branching rule.
        """
        self.branching_rule = branching_rule
        self.
