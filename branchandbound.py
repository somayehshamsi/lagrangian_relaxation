import abc
from abc import abstractmethod
import random
import heapq

class BranchingRule(abc.ABC):
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
        #addcommennt
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

class Node(abc.ABC):
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

        # Statistics
        self.total_nodes_solved = 0
        self.nodes_pruned_lower_bound = 0
        self.nodes_pruned_feasible = 0
        self.nodes_pruned_invalid_mst = 0

    def solve(self, root: Node):
        heapq.heappush(self.priority_queue, (root.local_lower_bound, root))
        node_counter = 0

        while self.priority_queue:
            _, node = heapq.heappop(self.priority_queue)
            self.total_nodes_solved += 1
            node_counter += 1
            
            if node_counter <= 10:
                print(f"\n--- Node {node_counter} ---")
                print(f"Fixed edges: {node.fixed_edges}")
                print(f"Excluded edges: {node.excluded_edges}")
                print(f"MST edges: {node.mst_edges}")
                print(f"Lower bound: {node.local_lower_bound}")
                print(f"Upper bound: {self.best_upper_bound}")

            # Prune if the node's bound is worse than the best found
            if node.local_lower_bound >= self.best_upper_bound:
                self.nodes_pruned_lower_bound += 1
                if node_counter <= 10:
                    print("Decision: Prune (lower bound >= best upper bound)")
                continue

            # Check feasibility
            is_feasible, reason = node.is_feasible()

            if is_feasible:

                # If feasible, update the best upper bound and prune the node
                upper_bound = node.compute_upper_bound()
                if upper_bound < self.best_upper_bound:
                    self.best_upper_bound = upper_bound
                    self.best_solution = node


                # If feasible, we check if the lower bound is less than the upper bound
                if node.local_lower_bound < self.best_upper_bound:
                    # If lower bound is less, continue branching like we do for "MST length exceeds budget"
                    candidates = node.get_branching_candidates()
                    if not candidates:
                        if node_counter <= 10:
                            print("Decision: Prune (no candidates for branching)")
                        continue
                    branching_object = self.branching_rule.get_branching_variable(candidates)
                    children = node.create_children(branching_object)

                    for child in children:
                        if child.local_lower_bound < self.best_upper_bound:
                            heapq.heappush(self.priority_queue, (child.local_lower_bound, child))
                    if node_counter <= 10:
                        print(f"Decision: Continue branching (lower bound < upper bound, children added to queue)")

                else:
                    if node_counter <= 10:
                        print(f"Decision: Prune (feasible solution found with upper bound: {upper_bound})")
                        self.nodes_pruned_feasible += 1
                        continue

            else:
                # If infeasible due to budget, branch further
                if reason == "MST length exceeds budget":
                    candidates = node.get_branching_candidates()
                    if not candidates:
                        if node_counter <= 10:
                            print("Decision: Prune (no candidates for branching)")
                        continue
                    branching_object = self.branching_rule.get_branching_variable(candidates)
                    children = node.create_children(branching_object)

                    for child in children:
                        if child.local_lower_bound < self.best_upper_bound:
                            heapq.heappush(self.priority_queue, (child.local_lower_bound, child))
                    if node_counter <= 10:
                        print(f"Decision: Branch (child added to queue with lower bound: {child.local_lower_bound})")
                else:
                    # If infeasible due to connectivity or missing nodes, prune the node
                    self.nodes_pruned_invalid_mst += 1
                    if node_counter <= 10:
                        print(f"Decision: Prune ({reason})")

                    continue

        # Print statistics
        print("\n--- Statistics ---")
        print(f"Total nodes solved: {self.total_nodes_solved}")
        print(f"Nodes pruned due to lower bound: {self.nodes_pruned_lower_bound}")
        print(f"Nodes pruned due to feasible solution: {self.nodes_pruned_feasible}")
        print(f"Nodes pruned due to invalid MST: {self.nodes_pruned_invalid_mst}")

        return self.best_solution, self.best_upper_bound

    
    @abc.abstractmethod
    def get_branching_candidates(self, node: Node):
        """
        Returns the candidates for branching from a node.
        """
        pass