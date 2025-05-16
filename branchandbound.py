    
import abc
from abc import abstractmethod
import random
import heapq
import networkx as nx
import math
import csv
from datetime import datetime
from typing import Optional, List, Tuple
import os
import time

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
        self.deleted.add(item)

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
    def __init__(self, branching_rule: BranchingRule, verbose=False, duality_gap_threshold=0.001, 
                 stagnation_limit=1000, config: dict = None, instance_seed: int = None):
        """
        Initializes the B&B with a branching rule, verbose flag, duality gap threshold, stagnation limit,
        configuration details, and instance seed.

        Args:
            branching_rule: The branching rule to use.
            verbose: Whether to print verbose output.
            duality_gap_threshold: Threshold for duality gap to stop branching.
            stagnation_limit: Limit for nodes without improvement before reducing max_nodes.
            config: Dictionary containing configuration details (e.g., branching_rule, use_2opt, use_shooting, use_bisection).
            instance_seed: Seed used to generate the instance.
        """
        self.branching_rule = branching_rule
        self.verbose = verbose
        self.duality_gap_threshold = duality_gap_threshold
        self.stagnation_limit = stagnation_limit
        self.best_upper_bound = float("inf")
        self.best_lower_bound = float("-inf")
        self.all_lower_bounds = []
        self.pruned_lower_bounds = []
        self.best_solution = None
        self.priority_queue = LazyPriorityQueue()

        # Configuration and instance details
        self.config = config or {}
        self.instance_seed = instance_seed

        # Generate a unique config identifier for the file name
        config_str = (f"{self.config.get('branching_rule', 'unknown')}_")
        seed_str = f"seed_{self.instance_seed}" if self.instance_seed is not None else "noseed"
        nf_path = os.path.join(os.path.expanduser("~/Desktop"), "nf")
        os.makedirs(nf_path, exist_ok=True)  # Create nf folder if it doesn't exist
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.node_log_file = os.path.join(nf_path, f"{config_str}_{seed_str}.csv")
        self._init_node_logger()

        # Statistics
        self.total_nodes_solved = 0
        self.nodes_pruned_lower_bound = 0
        self.nodes_pruned_feasible = 0
        self.nodes_pruned_invalid_mst = 0
        self.nodes_pruned_budget = 0
        self.nodes_pruned_gap = 0
        self.timed_out = False  # Track timeout status
        self.final_duality_gap = float("inf")  # Store final duality gap

    def _init_node_logger(self):
        headers = ["NodeID", "LowerBound", "UpperBound", "DualityGap", "BranchedVariable", 
                   "FixedEdges", "ExcludedEdges", "Reason", "Timestamp", "Config", "InstanceSeed"]
        with open(self.node_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def _log_node(self, node_id, node, branched_variable, reason, effective_upper_bound):
        duality_gap = effective_upper_bound - node.local_lower_bound if effective_upper_bound < float("inf") else float("inf")
        fixed_edges = str(list(node.fixed_edges)) if hasattr(node, 'fixed_edges') else "[]"
        excluded_edges = str(list(node.excluded_edges)) if hasattr(node, 'excluded_edges') else "[]"
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        config_str = (f"branching={self.config.get('branching_rule', 'unknown')},"
                      f"2opt={self.config.get('use_2opt', False)},"
                      f"shooting={self.config.get('use_shooting', False)},"
                      f"bisection={self.config.get('use_bisection', False)}")
        with open(self.node_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([node_id, node.local_lower_bound, effective_upper_bound, duality_gap, 
                            str(branched_variable), fixed_edges, excluded_edges, reason, timestamp,
                            config_str, self.instance_seed])

    def compute_initial_upper_bound(self, root: 'Node'):
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
        items = []
        for node in nodes:
            if math.isnan(node.local_lower_bound) or math.isinf(node.local_lower_bound):
                if self.verbose:
                    print(f"Skipping node with invalid lower bound: {node.local_lower_bound}")
                continue
            if node.local_lower_bound <= self.best_upper_bound:
                items.append((node.local_lower_bound, node))
                if self.verbose:
                    print(f"Adding node to queue with lower bound: {node.local_lower_bound}")
            else:
                if self.verbose:
                    print(f"Filtering node with lower bound {node.local_lower_bound} > best upper bound {self.best_upper_bound}")
        self.priority_queue.batch_push(items)

    def solve(self, root: Node):
        """
        Solves the Branch-and-Bound problem starting from the root node.
        
        Args:
            root: The root node of the B&B tree
        
        Returns:
            Tuple[Node, float]: Best solution and its upper bound
        """
        self.compute_initial_upper_bound(root)
        
        max_nodes = max(5000, 15000 - 50 * root.num_nodes)
        nodes_since_improvement = 0
        last_best_upper_bound = self.best_upper_bound

        self.priority_queue.push(root.local_lower_bound, root)
        node_counter = 0
        branched_variable = None
        start_time = time.time()

        while len(self.priority_queue) > 0 and node_counter < max_nodes:
            # Check for timeout
            if time.time() - start_time > 2000:
                self.timed_out = True
                if self.verbose:
                    print("Stopped: Exceeded 2000-second time limit")
                break

            try:
                min_lower_bound, node = self.priority_queue.pop()  # Always pop lowest
            except IndexError:
                break

            self.total_nodes_solved += 1
            node_counter += 1

            if math.isnan(node.local_lower_bound) or math.isinf(node.local_lower_bound):
                self._log_node(node_counter, node, branched_variable, "Invalid lower bound", self.best_upper_bound)
                if self.verbose:
                    print(f"Warning: Invalid lower bound {node.local_lower_bound} for node {node_counter}, skipping")
                continue
            self.all_lower_bounds.append(node.local_lower_bound)
            self.best_lower_bound = max(self.best_lower_bound, node.local_lower_bound)

            duality_gap = self.best_upper_bound - node.local_lower_bound if self.best_upper_bound < float("inf") else float("inf")
            threshold = min(self.duality_gap_threshold * self.best_upper_bound, 1.0) if self.best_upper_bound < float("inf") else float("inf")
            upper_bound = node.best_upper_bound
            if upper_bound < self.best_upper_bound:
                self.best_upper_bound = upper_bound
                self.best_solution = node
                nodes_since_improvement = 0

            if self.verbose:
                print(f"\n--- Node {node_counter} ---")
                print(f"Lower bound: {node.local_lower_bound}")
                print(f"Upper bound: {self.best_upper_bound}")
                print(f"Node duality gap: {duality_gap:.2f}")

            if node.local_lower_bound >= self.best_upper_bound:
                self.nodes_pruned_lower_bound += 1
                self._log_node(node_counter, node, branched_variable, "Lower bound >= best upper bound", self.best_upper_bound)
                self.priority_queue.mark_deleted(node)
                if self.verbose:
                    print("Decision: Prune (lower bound >= best upper bound)")
                continue

            if duality_gap < threshold:
                self.nodes_pruned_gap += 1
                self.pruned_lower_bounds.append(node.local_lower_bound)
                self._log_node(node_counter, node, branched_variable, "Duality gap too small", self.best_upper_bound)
                if self.verbose:
                    print(f"Debug: Appending lower bound {node.local_lower_bound} to pruned_lower_bounds")
                    print(f"Decision: Prune (duality gap {duality_gap:.2f} < {threshold:.2f})")
                self.priority_queue.mark_deleted(node)
                continue

            if self.best_upper_bound < float("inf"):
                threshold = min(self.duality_gap_threshold * self.best_upper_bound, 1.0)
                stop = True
                temp_heap = []
                while self.priority_queue.heap:
                    priority, count, item = heapq.heappop(self.priority_queue.heap)
                    if item not in self.priority_queue.deleted:
                        temp_heap.append((priority, count, item))
                        if self.best_upper_bound - priority >= threshold:
                            stop = False
                            break
                for priority, count, item in temp_heap:
                    heapq.heappush(self.priority_queue.heap, [priority, count, item])
                if stop and self.pruned_lower_bounds:
                    min_pruned_lower_bound = min(self.pruned_lower_bounds)
                    final_duality_gap = self.best_upper_bound - min_pruned_lower_bound if self.best_upper_bound < float("inf") else float("inf")
                    if self.verbose:
                        print(f"Stopping early: No remaining nodes with duality gap >= {threshold:.2f}")
                        print(f"Final duality gap: {final_duality_gap:.2f}")
                    break

            is_feasible, reason = node.is_feasible()

            if is_feasible:
                if node.local_lower_bound < self.best_upper_bound:
                    candidates = node.get_branching_candidates()
                    if not candidates:
                        self._log_node(node_counter, node, branched_variable, "No branching candidates", self.best_upper_bound)
                        if self.verbose:
                            print("Decision: Prune (no candidates for branching)")
                        continue
                    branched_variable = self.branching_rule.get_branching_variable(candidates)
                    children = node.create_children(branched_variable)
                    self.batch_insert_nodes(children)
                    for child in children:
                        if hasattr(child, 'is_child_likely_feasible') and not child.is_child_likely_feasible():
                            self.nodes_pruned_budget += 1
                            self._log_node(node_counter, child, branched_variable, "Child budget violation", self.best_upper_bound)
                            self.priority_queue.mark_deleted(child)
                            if self.verbose:
                                print("Decision: Prune child (budget violation likely)")
                    self._log_node(node_counter, node, branched_variable, "Branching", self.best_upper_bound)
                    if self.verbose:
                        print(f"Decision: Continue branching (lower bound < upper bound, children added to queue)")
                else:
                    self.nodes_pruned_feasible += 1
                    self._log_node(node_counter, node, branched_variable, "Feasible solution", self.best_upper_bound)
                    if self.verbose:
                        print(f"Decision: Prune (feasible solution found with upper bound: {upper_bound})")
                    self.priority_queue.mark_deleted(node)
                    continue
            else:
                if reason == "MST length exceeds budget":
                    candidates = node.get_branching_candidates()
                    if not candidates:
                        self._log_node(node_counter, node, branched_variable, "No branching candidates", self.best_upper_bound)
                        if self.verbose:
                            print("Decision: Prune (no candidates for branching)")
                        continue
                    branched_variable = self.branching_rule.get_branching_variable(candidates)
                    children = node.create_children(branched_variable)
                    self.batch_insert_nodes(children)
                    for child in children:
                        if hasattr(child, 'is_child_likely_feasible') and not child.is_child_likely_feasible():
                            self.nodes_pruned_budget += 1
                            self._log_node(node_counter, child, branched_variable, "Child budget violation", self.best_upper_bound)
                            self.priority_queue.mark_deleted(child)
                            if self.verbose:
                                print("Decision: Prune child (budget violation likely)")
                    self._log_node(node_counter, node, branched_variable, "Branching", self.best_upper_bound)
                    if self.verbose:
                        print(f"Decision: Branch (children added to queue)")
                else:
                    self.nodes_pruned_invalid_mst += 1
                    self._log_node(node_counter, node, branched_variable, reason, self.best_upper_bound)
                    self.priority_queue.mark_deleted(node)
                    if self.verbose:
                        print(f"Decision: Prune ({reason})")
                    continue

            if nodes_since_improvement >= self.stagnation_limit and last_best_upper_bound == self.best_upper_bound:
                max_nodes = min(max_nodes, node_counter + 1000)
                if self.verbose:
                    print(f"Reducing max_nodes to {max_nodes} due to stagnation")

            last_best_upper_bound = self.best_upper_bound

        valid_pruned_bounds = [b for b in self.pruned_lower_bounds if not math.isnan(b) and not math.isinf(b)]
        valid_all_bounds = [b for b in self.all_lower_bounds if not math.isnan(b) and not math.isinf(b)]

        if valid_pruned_bounds:
            min_pruned_lower_bound = min(valid_pruned_bounds)
            self.final_duality_gap = self.best_upper_bound - min_pruned_lower_bound if self.best_upper_bound < float("inf") else float("inf")
        elif valid_all_bounds:
            min_lower_bound = min(valid_all_bounds)
            self.final_duality_gap = self.best_upper_bound - min_lower_bound if self.best_upper_bound < float("inf") else float("inf")
        else:
            self.final_duality_gap = float("inf")
            if self.verbose:
                print("Warning: No valid lower bounds available for final duality gap")

        if self.timed_out:
            self._log_node(node_counter, root, branched_variable, "Timeout after 2000 seconds", self.best_upper_bound)
            if self.verbose:
                print(f"Final duality gap on timeout: {self.final_duality_gap:.2f}")

        if node_counter >= max_nodes:
            if self.verbose:
                print(f"Stopped early: Reached maximum node limit ({max_nodes})")

        print("\n--- Statistics ---")
        print(f"Total nodes solved: {self.total_nodes_solved}")
        print(f"Nodes pruned due to lower bound: {self.nodes_pruned_lower_bound}")
        print(f"Nodes pruned due to feasible solution: {self.nodes_pruned_feasible}")
        print(f"Nodes pruned due to invalid MST: {self.nodes_pruned_invalid_mst}")
        print(f"Nodes pruned due to budget violation: {self.nodes_pruned_budget}")
        print(f"Nodes pruned due to duality gap: {self.nodes_pruned_gap}")
        print(f"Final duality gap: {self.final_duality_gap:.2f}")
        if self.timed_out:
            print("Process stopped due to timeout (2000 seconds)")

        return self.best_solution, self.best_upper_bound

    def solve_with_shooting(self, root: Node, initial_lower_bound: float, initial_upper_bound: float, initial_solution: Optional[List[Tuple[int, int]]]):
        """
        Solves the Branch-and-Bound problem using the shooting method, as described in Yamada et al.
        
        Args:
            root: The root node of the B&B tree
            initial_lower_bound: Initial lower bound (e.g., from Lagrangian or 2-opt)
            initial_upper_bound: Initial upper bound (e.g., from Lagrangian or 2-opt)
            initial_solution: Initial solution edges (e.g., from 2-opt or None)
        
        Returns:
            Tuple[Node, float]: Best solution and its upper bound
        """
        from mstkpbranchandbound import MSTNode  # Import here to avoid circular import

        alpha = 0.3  # Parameter from the paper
        lower_bound = initial_lower_bound
        upper_bound = initial_upper_bound if initial_upper_bound < float("inf") else math.floor(root.best_upper_bound)
        
        # Initialize best solution with initial_solution if provided
        if initial_solution:
            self.best_solution = MSTNode(
                root.edges,
                root.num_nodes,
                root.budget,
                fixed_edges=set(initial_solution)
            )
            self.best_upper_bound = initial_upper_bound
        else:
            self.best_solution = None
            self.best_upper_bound = upper_bound

        shooting_trials = 0
        start_time = time.time()

        while lower_bound < upper_bound:
            # Check for timeout
            if time.time() - start_time > 2000:
                self.timed_out = True
                if self.verbose:
                    print("Shooting method stopped: Exceeded 2000-second time limit")
                break

            shooting_trials += 1
            incumbent_guess = math.floor(alpha * lower_bound + (1 - alpha) * upper_bound)
            if self.verbose:
                print(f"\nShooting Method Trial {shooting_trials}: Guessing incumbent = {incumbent_guess}")
                print(f"Lower bound = {lower_bound}, Upper bound = {upper_bound}")

            # Reset B&B state for this trial
            self.priority_queue = LazyPriorityQueue()
            self.total_nodes_solved = 0
            self.nodes_pruned_lower_bound = 0
            self.nodes_pruned_feasible = 0
            self.nodes_pruned_invalid_mst = 0
            self.nodes_pruned_budget = 0
            self.nodes_pruned_gap = 0
            self.all_lower_bounds = []
            self.pruned_lower_bounds = []
            self.best_lower_bound = float("-inf")
            self.timed_out = False  # Reset timeout flag for this trial

            # Run B&B with the guessed incumbent
            solution, upper_bound = self.solve(root)

            if self.timed_out:
                self._log_node(0, root, None, f"Shooting trial {shooting_trials} timeout after 2000 seconds", self.best_upper_bound)
                break

            if solution and upper_bound <= incumbent_guess:
                if self.verbose:
                    print(f"Shooting method succeeded after {shooting_trials} trials.")
                return solution, upper_bound
            else:
                # Update upper bound and try again
                upper_bound = incumbent_guess
                if self.verbose:
                    print(f"Shooting method failed. Updating upper bound to {upper_bound}.")

        if self.timed_out:
            self._log_node(0, root, None, "Shooting method timeout after 2000 seconds", self.best_upper_bound)
            if self.verbose:
                print(f"Final duality gap on timeout: {self.final_duality_gap:.2f}")

        if self.verbose:
            print(f"Shooting method converged to bounds: Lower = {lower_bound}, Upper = {upper_bound}")
        return self.best_solution, self.best_upper_bound
