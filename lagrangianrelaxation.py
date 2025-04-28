import networkx as nx
import numpy as np
import random
from time import time
from collections import defaultdict, OrderedDict

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n  # Added for size-based union
    
    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])  # Full path compression
        return self.parent[u]
    
    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        if pu == pv:
            return False
        if self.size[pu] < self.size[pv]:
            pu, pv = pv, pu
        self.parent[pv] = pu
        self.size[pu] += self.size[pv]
        self.rank[pu] = max(self.rank[pu], self.rank[pv] + 1)
        return True
    
    def connected(self, u, v):
        return self.find(u) == self.find(v)
    
    def count_components(self):
        return len(set(self.find(i) for i in range(len(self.parent))))

class LRUCache:
    """A simple LRU cache implementation using OrderedDict."""
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity
    
    def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key, value):
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

class LagrangianMST:
    total_compute_time = 0

    def __init__(self, edges, num_nodes, budget, fixed_edges=None, excluded_edges=None, 
                 initial_lambda=0.1, step_size=0.005, max_iter=300, p=0.95, 
                 use_cover_cuts=False, cut_frequency=5, use_bisection=False):
        start_time = time()
        self.edges = edges
        self.num_nodes = num_nodes
        self.budget = budget
        self.fixed_edges = set(fixed_edges) if fixed_edges else set()
        self.excluded_edges = set(excluded_edges) if excluded_edges else set()
        
        self.lmbda = initial_lambda
        self.step_size = step_size
        self.p = p
        self.max_iter = max_iter
        self.use_bisection = use_bisection

        self.best_lower_bound = float('-inf')
        self.best_upper_bound = float('inf')
        self.last_mst_edges = []
        self.primal_solutions = []
        self.step_sizes = []
        self.best_lambda = self.lmbda
        self.best_mst_edges = None
        self.best_cost = 0

        self.use_cover_cuts = use_cover_cuts
        self.cut_frequency = cut_frequency
        self.best_cuts = []
        self.best_cut_multipliers = {}

        # Optimized edge attribute caching
        self.edge_indices = {(min(u, v), max(u, v)): i for i, (u, v, _, _) in enumerate(edges)}
        self.edge_weights = np.array([w for _, _, w, _ in edges], dtype=float)
        self.edge_lengths = np.array([l for _, _, _, l in edges], dtype=float)
        self.modified_weights = np.array([w for _, _, w, _ in edges], dtype=float)  # Initial weights
        self.edge_list = [(u, v) for u, v, _, _ in edges]  # Store edge (u, v) pairs
        self.fixed_edge_indices = {self.edge_indices.get((min(u, v), max(u, v))) 
                                  for u, v in self.fixed_edges if (min(u, v), max(u, v)) in self.edge_indices}
        self.excluded_edge_indices = {self.edge_indices.get((min(u, v), max(u, v))) 
                                     for u, v in self.excluded_edges if (min(u, v), max(u, v)) in self.edge_indices}

        # Cache for MST results using LRU
        self.mst_cache = LRUCache(capacity=100)
        self.cache_tolerance = 1e-6  # Tolerance for cache key

        # Incremental update support
        self.last_modified_weights = None
        self.last_mst_edges = None

        # Graph for fallback or initialization
        self.graph = nx.Graph()
        self.graph.add_edges_from([(u, v) for u, v, _, _ in edges])
        self.edge_attributes = {(min(u, v), max(u, v)): (w, l) for u, v, w, l in edges}

        end_time = time()
        LagrangianMST.total_compute_time += end_time - start_time

    def generate_cover_cuts(self, mst_edges):
        # Unchanged for brevity; assumes mst_edges is a list of (u, v) tuples
        if not mst_edges:
            return []

        edge_data = self.edge_attributes
        total_length = sum(edge_data[(min(u, v), max(u, v))][1] for u, v in mst_edges)
        
        if total_length <= self.budget:
            return []

        fixed_edges_length = sum(
            edge_data[(min(u, v), max(u, v))][1]
            for u, v in self.fixed_edges
        )
        remaining_budget = self.budget - fixed_edges_length

        active_edges = [
            (u, v) for u, v in mst_edges
            if (u, v) not in self.fixed_edges and (v, u) not in self.fixed_edges
        ]

        cuts = []
        for i, (u1, v1) in enumerate(active_edges):
            l1 = edge_data[(min(u1, v1), max(u1, v1))][1]
            for j, (u2, v2) in enumerate(active_edges[i+1:], start=i+1):
                l2 = edge_data[(min(u2, v2), max(u2, v2))][1]
                if l1 + l2 > remaining_budget:
                    cuts.append({(u1, v1), (u2, v2)})

        for i, (u1, v1) in enumerate(active_edges):
            l1 = edge_data[(min(u1, v1), max(u1, v1))][1]
            for j, (u2, v2) in enumerate(active_edges[i+1:], start=i+1):
                l2 = edge_data[(min(u2, v2), max(u2, v2))][1]
                for k, (u3, v3) in enumerate(active_edges[j+1:], start=j+1):
                    l3 = edge_data[(min(u3, v3), max(u3, v3))][1]
                    if l1 + l2 + l3 > remaining_budget:
                        cuts.append({(u1, v1), (u2, v2), (u3, v3)})

        if len(mst_edges) >= 3:
            G = nx.Graph(mst_edges)
            degrees = dict(G.degree())
            hub_nodes = [n for n in degrees if degrees[n] > 2]
            for node in hub_nodes:
                neighbors = list(G.neighbors(node))
                for i in range(len(neighbors)):
                    for j in range(i + 1, len(neighbors)):
                        e1 = (node, neighbors[i])
                        e2 = (node, neighbors[j])
                        l1 = edge_data.get((min(e1[0], e1[1]), max(e1[0], e1[1])), (0, 0))[1]
                        l2 = edge_data.get((min(e2[0], e2[1]), max(e2[0], e2[1])), (0, 0))[1]
                        if l1 + l2 > remaining_budget:
                            cuts.append({e1, e2})

        if total_length > self.budget and len(mst_edges) >= 2:
            cuts.append(set(mst_edges))

        unique_cuts = []
        seen = set()
        for cut in cuts:
            frozen = frozenset(cut)
            if frozen not in seen:
                seen.add(frozen)
                unique_cuts.append(cut)
        
        return unique_cuts[:10]

    def compute_modified_weights(self):
        """Compute modified weights for all edges based on current lambda and cut multipliers."""
        modified_weights = self.edge_weights.copy()
        modified_weights += self.lmbda * self.edge_lengths
        
        if self.use_cover_cuts:
            for cut_idx, cut in enumerate(self.best_cuts):
                multiplier = self.best_cut_multipliers.get(cut_idx, 0)
                for u, v in cut:
                    edge_key = (min(u, v), max(u, v))
                    if edge_key in self.edge_indices:
                        edge_idx = self.edge_indices[edge_key]
                        modified_weights[edge_idx] += multiplier
        
        return modified_weights

    def custom_kruskal(self, modified_weights):
        """Custom Kruskal's algorithm incorporating fixed and excluded edges."""
        uf = UnionFind(self.num_nodes)
        mst_edges = []
        mst_cost = 0.0

        # Step 1: Include fixed edges
        for edge_idx in self.fixed_edge_indices:
            u, v = self.edge_list[edge_idx]
            if uf.union(u, v):
                mst_edges.append((u, v))
                mst_cost += modified_weights[edge_idx]
            else:
                # Fixed edges form a cycle, infeasible
                return float('inf'), float('inf'), []

        # Step 2: Sort edges by modified weights (excluding fixed/excluded)
        edge_indices = [i for i in range(len(self.edges)) 
                        if i not in self.fixed_edge_indices and i not in self.excluded_edge_indices]
        sorted_edges = sorted(edge_indices, key=lambda i: modified_weights[i])

        # Step 3: Add edges to MST
        for edge_idx in sorted_edges:
            u, v = self.edge_list[edge_idx]
            if uf.union(u, v):
                mst_edges.append((u, v))
                mst_cost += modified_weights[edge_idx]

        # Step 4: Verify MST
        if uf.count_components() > 1 or len(set(u for u, _ in mst_edges) | set(v for _, v in mst_edges)) < self.num_nodes:
            return float('inf'), float('inf'), []

        # Step 5: Compute real length
        mst_length = sum(self.edge_lengths[self.edge_indices[(min(u, v), max(u, v))]] 
                         for u, v in mst_edges)

        return mst_cost, mst_length, mst_edges

    def incremental_kruskal(self, prev_weights, prev_mst_edges, current_weights):
        """Incrementally update the MST based on weight changes."""
        uf = UnionFind(self.num_nodes)
        mst_edges = []
        mst_cost = 0.0

        # Step 1: Include fixed edges
        for edge_idx in self.fixed_edge_indices:
            u, v = self.edge_list[edge_idx]
            if uf.union(u, v):
                mst_edges.append((u, v))
                mst_cost += current_weights[edge_idx]
            else:
                return float('inf'), float('inf'), []

        # Step 2: Identify changed edges
        weight_changes = current_weights - prev_weights
        changed_indices = np.where(np.abs(weight_changes) > self.cache_tolerance)[0]
        changed_edges = set(changed_indices)

        # Step 3: Re-evaluate MST edges
        prev_mst_indices = {self.edge_indices[(min(u, v), max(u, v))] for u, v in prev_mst_edges
                            if self.edge_indices[(min(u, v), max(u, v))] not in self.fixed_edge_indices}
        candidate_indices = (prev_mst_indices | changed_edges) - self.excluded_edge_indices - self.fixed_edge_indices
        sorted_edges = sorted(candidate_indices, key=lambda i: current_weights[i])

        # Step 4: Add edges to MST
        for edge_idx in sorted_edges:
            u, v = self.edge_list[edge_idx]
            if uf.union(u, v):
                mst_edges.append((u, v))
                mst_cost += current_weights[edge_idx]

        # Step 5: Verify MST
        if uf.count_components() > 1 or len(set(u for u, _ in mst_edges) | set(v for _, v in mst_edges)) < self.num_nodes:
            return float('inf'), float('inf'), []

        # Step 6: Compute real length
        mst_length = sum(self.edge_lengths[self.edge_indices[(min(u, v), max(u, v))]] 
                         for u, v in mst_edges)

        return mst_cost, mst_length, mst_edges

    def compute_mst(self, modified_edges=None):
        """Compute MST using custom Kruskal's algorithm with caching."""
        start_time = time()
        
        # If modified_edges is provided, compute weights from scratch
        if modified_edges is not None:
            weights = np.array([w for _, _, w in modified_edges], dtype=float)
        else:
            weights = self.compute_modified_weights()

        # Create cache key with tolerance
        weights_key = tuple(np.round(weights / self.cache_tolerance).astype(int))

        # Check cache
        cached_result = self.mst_cache.get(weights_key)
        if cached_result is not None:
            end_time = time()
            LagrangianMST.total_compute_time += end_time - start_time
            return cached_result

        # Compute MST
        mst_cost, mst_length, mst_edges = self.custom_kruskal(weights)

        # Cache result
        self.mst_cache.put(weights_key, (mst_cost, mst_length, mst_edges))

        end_time = time()
        LagrangianMST.total_compute_time += end_time - start_time
        return mst_cost, mst_length, mst_edges

    def compute_mst_incremental(self, prev_weights, prev_mst_edges):
        """Compute MST incrementally by updating only changed weights."""
        current_weights = self.compute_modified_weights()
        weight_changes = current_weights - prev_weights

        # If no significant changes, reuse previous MST
        if np.all(np.abs(weight_changes) < 1e-4):  # Increased threshold
            mst_cost = sum(current_weights[self.edge_indices[(min(u, v), max(u, v))]] 
                           for u, v in prev_mst_edges)
            mst_length = sum(self.edge_lengths[self.edge_indices[(min(u, v), max(u, v))]] 
                             for u, v in prev_mst_edges)
            return mst_cost, mst_length, prev_mst_edges

        # Use incremental Kruskal for small changes
        return self.incremental_kruskal(prev_weights, prev_mst_edges, current_weights)

    def solve(self, inherited_cuts=None, inherited_multipliers=None):
        start_time = time()
        
        if inherited_cuts is not None:
            self.best_cuts = inherited_cuts
            self.best_cut_multipliers = inherited_multipliers.copy() if inherited_multipliers else {}
        
        prev_weights = None
        prev_mst_edges = None

        for iter_num in range(self.max_iter):
            # Use incremental MST computation if possible
            if prev_weights is not None and prev_mst_edges is not None:
                mst_cost, mst_length, mst_edges = self.compute_mst_incremental(prev_weights, prev_mst_edges)
            else:
                mst_cost, mst_length, mst_edges = self.compute_mst()

            self.last_mst_edges = mst_edges
            prev_mst_edges = mst_edges
            prev_weights = self.compute_modified_weights()

            is_feasible = mst_length <= self.budget
            self.primal_solutions.append((mst_edges, is_feasible))
            self.step_sizes.append(self.step_size)

            cover_cut_penalty = sum(
                multiplier * (len(cut) - 1)
                for cut_idx, cut in enumerate(self.best_cuts)
                for multiplier in [self.best_cut_multipliers.get(cut_idx, 0)]
            )

            lagrangian_bound = mst_cost - self.lmbda * self.budget - cover_cut_penalty

            if lagrangian_bound > self.best_lower_bound:
                self.best_lower_bound = lagrangian_bound
                self.best_lambda = self.lmbda
                self.best_mst_edges = mst_edges
                self.best_cost = mst_cost
                if self.use_cover_cuts and self.best_mst_edges and iter_num % self.cut_frequency == 0:
                    new_cuts = self.generate_cover_cuts(self.best_mst_edges)
                    for cut in new_cuts:
                        if not any(cut == existing for existing in self.best_cuts):
                            cut_idx = len(self.best_cuts)
                            self.best_cuts.append(cut)
                            self.best_cut_multipliers[cut_idx] = 1.0

            if is_feasible:
                uf = UnionFind(self.num_nodes)
                for u, v in mst_edges:
                    uf.union(u, v)
                if uf.count_components() == 1:
                    if mst_cost < self.best_upper_bound:
                        self.best_upper_bound = mst_cost
                        if self.use_cover_cuts:
                            new_cuts = self.generate_cover_cuts(mst_edges)
                            for cut in new_cuts:
                                if not any(cut == existing for existing in self.best_cuts):
                                    cut_idx = len(self.best_cuts)
                                    self.best_cuts.append(cut)
                                    self.best_cut_multipliers[cut_idx] = 1.0

            knapsack_subgradient = -(self.budget - mst_length)
            cut_subgradients = [
                sum(1 for e in mst_edges if e in cut or (e[1], e[0]) in cut) - (len(cut) - 1)
                for cut in self.best_cuts
            ]

            converged = (abs(knapsack_subgradient) < 1e-5 and 
                         all(abs(g) < 1e-5 for g in cut_subgradients))
            duality_gap = self.best_upper_bound - self.best_lower_bound
            
            if converged or abs(duality_gap) < 1e-5:
                print(f"Converged! (Reason: {'Converged' if converged else 'Small duality gap'})")
                break

            self.step_size *= self.p
            self.lmbda = max(self.lmbda + self.step_size * knapsack_subgradient, 1e-2)
            if self.lmbda < 1e-6 and abs(knapsack_subgradient) > 1.0:
                self.lmbda = 0.1

            for cut_idx, violation in enumerate(cut_subgradients):
                current_mult = self.best_cut_multipliers.get(cut_idx, 0)
                new_mult = max(1e-4, current_mult + self.step_size * violation)
                if new_mult < 1e-6 and abs(violation) > 1.0:
                    new_mult = 0.1
                self.best_cut_multipliers[cut_idx] = min(new_mult, 10.0)

        end_time = time()
        LagrangianMST.total_compute_time += end_time - start_time

        new_cuts = []
        if self.use_cover_cuts and self.best_mst_edges:
            new_cuts = self.generate_cover_cuts(self.best_mst_edges)
            new_cuts = [cut for cut in new_cuts if not any(cut == existing for existing in self.best_cuts)]

        return self.best_lower_bound, self.best_upper_bound, new_cuts

    def compute_mst_for_lambda(self, lambda_val):
        """Compute MST for a specific lambda value."""
        modified_edges = []
        for i, (u, v) in enumerate(self.edge_list):
            modified_w = self.edge_weights[i] + lambda_val * self.edge_lengths[i]
            for cut_idx, cut in enumerate(self.best_cuts):
                if (u, v) in cut or (v, u) in cut:
                    modified_w += self.best_cut_multipliers.get(cut_idx, 0)
            modified_edges.append((u, v, modified_w))
        return self.compute_mst(modified_edges)

    def compute_shor_primal_solution(self):
        if not self.primal_solutions:
            return None
        edge_weights = {}
        total_weight = 0.0
        for i, (mst_edges, _) in enumerate(self.primal_solutions):
            weight = self.step_sizes[i]
            total_weight += weight
            for edge in mst_edges:
                edge_weights[edge] = edge_weights.get(edge, 0) + weight
        if total_weight > 0:
            for edge in edge_weights:
                edge_weights[edge] /= total_weight
        return edge_weights

    def compute_dantzig_wolfe_solution(self):
        if len(self.primal_solutions) < 2:
            return None
        feasible = [mst for mst, feasible in self.primal_solutions if feasible]
        infeasible = [mst for mst, feasible in self.primal_solutions if not feasible]
        if not feasible or not infeasible:
            return None
        mst_feas = feasible[-1]
        mst_infeas = infeasible[-1]
        def get_length(mst):
            return sum(self.edge_lengths[self.edge_indices[(min(u, v), max(u, v))]] 
                       for u, v in mst)
        len_feas = get_length(mst_feas)
        len_infeas = get_length(mst_infeas)
        try:
            α = (self.budget - len_feas) / (len_infeas - len_feas)
            α = max(0, min(1, α))
        except ZeroDivisionError:
            return None
        combined = {}
        all_edges = set(mst_feas) | set(mst_infeas)
        for e in all_edges:
            in_feas = e in mst_feas or (e[1], e[0]) in mst_feas
            in_infeas = e in mst_infeas or (e[1], e[0]) in mst_infeas
            if in_feas and in_infeas:
                combined[e] = 1.0
            elif in_infeas:
                combined[e] = α
            elif in_feas:
                combined[e] = 1 - α
            else:
                combined[e] = 0.0
        return combined

    def compute_real_weight_length(self):
        real_weight = sum(self.edge_weights[self.edge_indices[(min(u, v), max(u, v))]] 
                          for u, v in self.last_mst_edges)
        real_length = sum(self.edge_lengths[self.edge_indices[(min(u, v), max(u, v))]] 
                          for u, v in self.last_mst_edges)
        return real_weight, real_length
