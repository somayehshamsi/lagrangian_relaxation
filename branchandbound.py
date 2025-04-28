import abc
from abc import abstractmethod
import random
import heapq
import networkx as nx

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

MEANINGFUL_PENALTY = 1000  # Penalty for non-meaningful branches

class LazyPriorityQueue:
    """A priority queue with lazy deletion using heapq."""
    def __init__(self):
        self.heap = []
        self.entry_count = 0  # To handle ties in heap
        self.deleted = set()  # Track deleted entries

    def push(self, priority, item):
        """Push an item with given priority."""
        if item not in self.deleted:
            entry = [priority, self.entry_count, item]
            self.entry_count += 1
            heapq.heappush(self.heap, entry)

    def pop(self):
        """Pop the item with the lowest priority, skipping deleted items."""
        while self.heap:
            priority, count, item = heapq.heappop(self.heap)
            self.deleted.discard(item)  # Clean up deleted set
            if item not in self.deleted:
                return priority, item
        raise IndexError("pop from an empty priority queue")

    def mark_deleted(self, item):
        """Mark an item as deleted without removing it immediately."""
        self.deleted.add(item)  # No check needed since set ensures uniqueness

    def batch_push(self, items):
        """Push multiple items at once."""
        for priority, item in items:
            if item not in self.deleted:
                entry = [priority, self.entry_count, item]
                self.entry_count += 1
                heapq.heappush(self.heap, entry)

    def __len__(self):
        """Return the number of non-deleted items (approximate, non-negative)."""
        return max(0, len(self.heap) - len(self.deleted))

class BranchAndBound:
    """
    An implementation of the Branch-and-Bound
    """
    def __init__(self, branching_rule: BranchingRule, verbose=False, duality_gap_threshold=0.001, stagnation_limit=1000):
        """
        Initializes the B&B with a branching rule, verbose flag, duality gap threshold, and stagnation limit.
        """
        self.branching_rule = branching_rule
        self.verbose = verbose
        self.duality_gap_threshold = duality_gap_threshold  # Relative gap threshold
        self.stagnation_limit = stagnation_limit  # Nodes without improvement before reducing max_nodes
        self.best_upper_bound = float("inf")
        self.best_solution = None
        self.priority_queue = LazyPriorityQueue()  # Use custom priority queue

        # Statistics
        self.total_nodes_solved = 0
        self.nodes_pruned_lower_bound = 0
        self.nodes_pruned_feasible = 0
        self.nodes_pruned_invalid_mst = 0
        self.nodes_pruned_budget = 0

    def compute_initial_upper_bound(self, root: 'MSTNode'):
        """
        Computes an initial upper bound using a greedy MST heuristic.
        """
        G = nx.Graph()
        G.add_weighted_edges_from([(u, v, w) for u, v, w, l in root.edges])
        mst = nx.minimum_spanning_tree(G)
        total_length = sum(root.lagrangian_solver.edge_attributes[(min(u, v), max(u, v))][1] for u, v in mst.edges)
        if total_length <= root.budget:
            total_weight = sum(root.lagrangian_solver.edge_attributes[(min(u, v), max(u, v))][0] for u, v in mst.edges)
            self.best_upper_bound = total_weight
            self.best_solution = root
            root.mst_edges = list(mst.edges)
            if self.verbose:
                print(f"Initial feasible MST found with weight: {total_weight}, length: {total_length}")

    def batch_insert_nodes(self, nodes):
        """
        Batch insert nodes into the priority queue, filtering out those with high lower bounds.
        """
        items = [(node.local_lower_bound, node) for node in nodes
                 if node.local_lower_bound < self.best_upper_bound]
        self.priority_queue.batch_push(items)

    def solve(self, root: Node):
        """
        Solves the Branch-and-Bound problem starting from the root node.
        """
        self.compute_initial_upper_bound(root)
        
        max_nodes = max(2000, 10000 - 50 * root.num_nodes)  # e.g., 7500 for num_nodes=50
        nodes_since_improvement = 0
        last_best_upper_bound = self.best_upper_bound

        self.priority_queue.push(root.local_lower_bound, root)
        node_counter = 0

        while len(self.priority_queue) > 0 and node_counter < max_nodes:
            try:
                min_lower_bound, node = self.priority_queue.pop()
            except IndexError:
                break

            self.total_nodes_solved += 1
            node_counter += 1

            # if self.verbose and node_counter <= 100:
            print(f"\n--- Node {node_counter} ---")
            print(f"Lower bound: {node.local_lower_bound}")
            print(f"Upper bound: {self.best_upper_bound}")

            if node.local_lower_bound >= self.best_upper_bound:
                self.nodes_pruned_lower_bound += 1
                self.priority_queue.mark_deleted(node)
                if self.verbose and node_counter <= 100:
                    print("Decision: Prune (lower bound >= best upper bound)")
                continue

            # Check duality gap
            if self.best_upper_bound < float("inf") and len(self.priority_queue) > 0:
                # Peek at the next lower bound
                try:
                    next_lower_bound, next_node = self.priority_queue.pop()
                    self.priority_queue.push(next_lower_bound, next_node)
                    duality_gap = self.best_upper_bound - next_lower_bound
                    if duality_gap < min(self.duality_gap_threshold * self.best_upper_bound, 1.0):
                        print(f"Stopping early: Duality gap {duality_gap:.2f} < "
                              f"{min(self.duality_gap_threshold * self.best_upper_bound, 1.0):.2f}")
                        break
                except IndexError:
                    pass

            is_feasible, reason = node.is_feasible()

            if is_feasible:
                upper_bound = node.compute_upper_bound()
                if upper_bound < self.best_upper_bound:
                    self.best_upper_bound = upper_bound
                    self.best_solution = node
                    nodes_since_improvement = 0
                else:
                    nodes_since_improvement += 1

                if node.local_lower_bound < self.best_upper_bound:
                    candidates = node.get_branching_candidates()
                    if not candidates:
                        if self.verbose and node_counter <= 100:
                            print("Decision: Prune (no candidates for branching)")
                        continue
                    branching_object = self.branching_rule.get_branching_variable(candidates)
                    children = node.create_children(branching_object)

                    # Batch insert children
                    self.batch_insert_nodes(children)
                    for child in children:
                        if hasattr(child, 'is_child_likely_feasible') and not child.is_child_likely_feasible():
                            self.nodes_pruned_budget += 1
                            self.priority_queue.mark_deleted(child)
                            if self.verbose and node_counter <= 100:
                                print("Decision: Prune child (budget violation likely)")
                    if self.verbose and node_counter <= 100:
                        print(f"Decision: Continue branching (lower bound < upper bound, children added to queue)")

                else:
                    if self.verbose and node_counter <= 100:
                        print(f"Decision: Prune (feasible solution found with upper bound: {upper_bound})")
                    self.nodes_pruned_feasible += 1
                    self.priority_queue.mark_deleted(node)
                    continue

            else:
                if reason == "MST length exceeds budget":
                    candidates = node.get_branching_candidates()
                    if not candidates:
                        if self.verbose and node_counter <= 100:
                            print("Decision: Prune (no candidates for branching)")
                        continue
                    branching_object = self.branching_rule.get_branching_variable(candidates)
                    children = node.create_children(branching_object)

                    # Batch insert children
                    self.batch_insert_nodes(children)
                    for child in children:
                        if hasattr(child, 'is_child_likely_feasible') and not child.is_child_likely_feasible():
                            self.nodes_pruned_budget += 1
                            self.priority_queue.mark_deleted(child)
                            if self.verbose and node_counter <= 100:
                                print("Decision: Prune child (budget violation likely)")
                    if self.verbose and node_counter <= 100:
                        print(f"Decision: Branch (children added to queue)")

                else:
                    self.nodes_pruned_invalid_mst += 1
                    self.priority_queue.mark_deleted(node)
                    if self.verbose and node_counter <= 100:
                        print(f"Decision: Prune ({reason})")
                    continue

            if nodes_since_improvement >= self.stagnation_limit and last_best_upper_bound == self.best_upper_bound:
                max_nodes = min(max_nodes, node_counter + 1000)
                if self.verbose:
                    print(f"Reducing max_nodes to {max_nodes} due to stagnation")

            last_best_upper_bound = self.best_upper_bound

        if node_counter >= max_nodes:
            print(f"Stopped early: Reached maximum node limit ({max_nodes})")

        print("\n--- Statistics ---")
        print(f"Total nodes solved: {self.total_nodes_solved}")
        print(f"Nodes pruned due to lower bound: {self.nodes_pruned_lower_bound}")
        print(f"Nodes pruned due to feasible solution: {self.nodes_pruned_feasible}")
        print(f"Nodes pruned due to invalid MST: {self.nodes_pruned_invalid_mst}")
        print(f"Nodes pruned due to budget violation: {self.nodes_pruned_budget}")
        print(f"Final duality gap: {(self.best_upper_bound - min_lower_bound) if len(self.priority_queue) > 0 else 0:.2f}")

        return self.best_solution, self.best_upper_bound

    @abc.abstractmethod
    def get_branching_candidates(self, node: Node):
        """
        Returns the candidates for branching from a node.
        """
        pass



