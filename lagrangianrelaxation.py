
import networkx as nx
import numpy as np
import random
from time import time
from collections import defaultdict, OrderedDict
from scipy.optimize import linprog

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n
    
    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
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
                 initial_lambda=0.1, step_size=0.01, max_iter=300, p=0.95, 
                 use_cover_cuts=False, cut_frequency=5, use_bisection=False, verbose=False):
        


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
        self.verbose = verbose

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

        self.multipliers = []

        self.edge_indices = {(min(u, v), max(u, v)): i for i, (u, v, _, _) in enumerate(edges)}
        self.edge_weights = np.array([w for _, _, w, _ in edges], dtype=float)
        self.edge_lengths = np.array([l for _, _, _, l in edges], dtype=float)
        self.modified_weights = np.array([w for _, _, w, _ in edges], dtype=float)
        self.edge_list = [(u, v) for u, v, _, _ in edges]
        self.fixed_edge_indices = {self.edge_indices.get((min(u, v), max(u, v))) 
                                  for u, v in self.fixed_edges if (min(u, v), max(u, v)) in self.edge_indices}
        self.excluded_edge_indices = {self.edge_indices.get((min(u, v), max(u, v))) 
                                     for u, v in self.excluded_edges if (min(u, v), max(u, v)) in self.edge_indices}

        self.mst_cache = LRUCache(capacity=100)
        self.cache_tolerance = 1e-8

        self.last_modified_weights = None
        self.last_mst_edges = None

        self.graph = nx.Graph()
        self.graph.add_edges_from([(u, v) for u, v, _, _ in edges])
        self.edge_attributes = {(min(u, v), max(u, v)): (w, l) for u, v, w, l in edges}

        end_time = time()
        LagrangianMST.total_compute_time += end_time - start_time

    def generate_cover_cuts(self, mst_edges):
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
        if not active_edges:
            return []

        cuts = []

        edge_ratios = [
            (e, edge_data[(min(e[0], e[1]), max(e[0], e[1]))][1] / edge_data[(min(e[0], e[1]), max(e[0], e[1]))][0])
            for e in active_edges
        ]
        edge_ratios.sort(key=lambda x: x[1], reverse=True)

        current_edges = [e for e, _ in edge_ratios]
        current_length = sum(edge_data[(min(e[0], e[1]), max(e[0], e[1]))][1] for e in current_edges)

        # === Andreas' suggestion: Max active-edge cut over budget ===
        sorted_by_length = sorted(active_edges, key=lambda e: edge_data[(min(e[0], e[1]), max(e[0], e[1]))][1])
        max_violation_cut = []
        total = 0
        for edge in sorted_by_length:
            edge_len = edge_data[(min(edge[0], edge[1]), max(edge[0], edge[1]))][1]
            if total + edge_len <= remaining_budget:
                total += edge_len
                continue
            max_violation_cut.append(edge)
            total += edge_len
        if max_violation_cut:
            cuts.append(set(max_violation_cut))
        # === End of Andreas' suggestion ===

        # Minimal cover logic
        minimal_cover = current_edges.copy()
        minimal_length = current_length
        for edge, _ in edge_ratios:
            edge_key = (min(edge[0], edge[1]), max(edge[0], edge[1]))
            length = edge_data[edge_key][1]
            if minimal_length - length > remaining_budget and len(minimal_cover) > 2:
                minimal_cover.remove(edge)
                minimal_length -= length
            else:
                break

        if minimal_length > remaining_budget and minimal_cover:
            cover_set = set(minimal_cover)
            cuts.append(cover_set)

        # Pair-wise cuts
        pair_candidates = []
        for i, (u1, v1) in enumerate(active_edges):
            l1 = edge_data[(min(u1, v1), max(u1, v1))][1]
            for j, (u2, v2) in enumerate(active_edges[i+1:], start=i+1):
                l2 = edge_data[(min(u2, v2), max(u2, v2))][1]
                pair_cut = {(u1, v1), (u2, v2)}
                pair_length = l1 + l2
                pair_candidates.append((pair_cut, pair_length))
                if pair_length > remaining_budget:
                    cuts.append(pair_cut)

        pair_candidates.sort(key=lambda x: x[1], reverse=True)
        for pair_cut, total_length in pair_candidates[:5]:
            if pair_cut not in cuts:
                cuts.append(pair_cut)

        # Triplet cuts
        triplet_count = 0
        for i, (u1, v1) in enumerate(active_edges):
            l1 = edge_data[(min(u1, v1), max(u1, v1))][1]
            for j, (u2, v2) in enumerate(active_edges[i+1:], start=i+1):
                l2 = edge_data[(min(u2, v2), max(u2, v2))][1]
                for k, (u3, v3) in enumerate(active_edges[j+1:], start=j+1):
                    l3 = edge_data[(min(u3, v3), max(u3, v3))][1]
                    if l1 + l2 + l3 > remaining_budget:
                        triplet_cut = {(u1, v1), (u2, v2), (u3, v3)}
                        cuts.append(triplet_cut)
                        triplet_count += 1
                        if triplet_count >= 10:
                            break
                if triplet_count >= 10:
                    break
            if triplet_count >= 10:
                break

        # Hub-node cuts
        if len(mst_edges) >= 3:
            G = nx.Graph(mst_edges)
            degrees = dict(G.degree())
            hub_nodes = [n for n in degrees if degrees[n] > 2]
            for node in hub_nodes:
                neighbors = list(G.neighbors(node))
                for i in range(len(neighbors)):
                    for j in range(i + 1, len(neighbors)):
                        e1 = (node, neighbors[i]) if (node, neighbors[i]) in edge_data else (neighbors[i], node)
                        e2 = (node, neighbors[j]) if (node, neighbors[j]) in edge_data else (neighbors[j], node)
                        l1 = edge_data.get((min(e1[0], e1[1]), max(e1[0], e1[1])), (0, 0))[1]
                        l2 = edge_data.get((min(e2[0], e2[1]), max(e2[0], e2[1])), (0, 0))[1]
                        if l1 + l2 > remaining_budget:
                            hub_cut = {e1, e2}
                            cuts.append(hub_cut)

        # Full MST cut
        if total_length > self.budget and len(mst_edges) <= 60:
            full_mst_cut = set(mst_edges)
            cuts.append(full_mst_cut)

        # Filter and deduplicate cuts
        filtered_cuts = []
        for cut in cuts:
            cut_sum = sum(edge_data[(min(u, v), max(u, v))][1] for u, v in cut)
            violation = cut_sum - remaining_budget
            if violation > 0 and len(cut) <= 60:
                filtered_cuts.append(cut)

        cut_violations = [
            (cut, sum(edge_data[(min(u, v), max(u, v))][1] for u, v in cut) - remaining_budget)
            for cut in filtered_cuts
        ]
        cut_violations.sort(key=lambda x: x[1], reverse=True)
        unique_cuts = []
        seen = set()
        for cut, _ in cut_violations:
            frozen = frozenset(cut)
            if frozen not in seen:
                seen.add(frozen)
                unique_cuts.append(cut)

        return unique_cuts[:10]

    def compute_modified_weights(self):
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
        uf = UnionFind(self.num_nodes)
        mst_edges = []
        mst_cost = 0.0

        for edge_idx in self.fixed_edge_indices:
            u, v = self.edge_list[edge_idx]
            if uf.union(u, v):
                mst_edges.append((u, v))
                mst_cost += modified_weights[edge_idx]
            else:
                return float('inf'), float('inf'), []

        edge_indices = [i for i in range(len(self.edges)) 
                        if i not in self.fixed_edge_indices and i not in self.excluded_edge_indices]
        sorted_edges = sorted(edge_indices, key=lambda i: modified_weights[i])

        for edge_idx in sorted_edges:
            u, v = self.edge_list[edge_idx]
            if uf.union(u, v):
                mst_edges.append((u, v))
                mst_cost += modified_weights[edge_idx]

        if uf.count_components() > 1 or len(set(u for u, _ in mst_edges) | set(v for _, v in mst_edges)) < self.num_nodes:
            return float('inf'), float('inf'), []

        mst_length = sum(self.edge_lengths[self.edge_indices[(min(u, v), max(u, v))]] 
                         for u, v in mst_edges)

        return mst_cost, mst_length, mst_edges

    def incremental_kruskal(self, prev_weights, prev_mst_edges, current_weights):
        uf = UnionFind(self.num_nodes)
        mst_edges = []
        mst_cost = 0.0

        for edge_idx in self.fixed_edge_indices:
            u, v = self.edge_list[edge_idx]
            if uf.union(u, v):
                mst_edges.append((u, v))
                mst_cost += current_weights[edge_idx]
            else:
                return float('inf'), float('inf'), []

        weight_changes = current_weights - prev_weights
        changed_indices = np.where(np.abs(weight_changes) > self.cache_tolerance)[0]
        changed_edges = set(changed_indices)

        prev_mst_indices = {self.edge_indices[(min(u, v), max(u, v))] for u, v in prev_mst_edges
                            if self.edge_indices[(min(u, v), max(u, v))] not in self.fixed_edge_indices}
        candidate_indices = (prev_mst_indices | changed_edges) - self.excluded_edge_indices - self.fixed_edge_indices
        sorted_edges = sorted(candidate_indices, key=lambda i: current_weights[i])

        for edge_idx in sorted_edges:
            u, v = self.edge_list[edge_idx]
            if uf.union(u, v):
                mst_edges.append((u, v))
                mst_cost += current_weights[edge_idx]

        if uf.count_components() > 1 or len(set(u for u, _ in mst_edges) | set(v for _, v in mst_edges)) < self.num_nodes:
            return float('inf'), float('inf'), []

        mst_length = sum(self.edge_lengths[self.edge_indices[(min(u, v), max(u, v))]] 
                         for u, v in mst_edges)

        return mst_cost, mst_length, mst_edges

    def compute_mst(self, modified_edges=None):
        start_time = time()
        
        if modified_edges is not None:
            weights = np.array([w for _, _, w in modified_edges], dtype=float)
        else:
            weights = self.compute_modified_weights()

        weights_key = tuple(np.round(weights / self.cache_tolerance).astype(int))

        self.consecutive_cache_hits = getattr(self, 'consecutive_cache_hits', 0)
        cached_result = self.mst_cache.get(weights_key)
        if cached_result is not None and self.consecutive_cache_hits < 5:
            self.consecutive_cache_hits += 1
            if self.verbose:
                print(f"Cache hit: Retrieved MST with length={cached_result[1]:.2f}, consecutive_hits={self.consecutive_cache_hits}")
            end_time = time()
            LagrangianMST.total_compute_time += end_time - start_time
            return cached_result
        
        self.consecutive_cache_hits = 0
        if self.verbose:
            print(f"Cache miss: Computing new MST for weights_key")
        
        mst_cost, mst_length, mst_edges = self.custom_kruskal(weights)
        if self.verbose:
            print(f"New MST computed: length={mst_length:.2f}")
        
        self.mst_cache.put(weights_key, (mst_cost, mst_length, mst_edges))

        end_time = time()
        LagrangianMST.total_compute_time += end_time - start_time
        return mst_cost, mst_length, mst_edges

    def compute_mst_incremental(self, prev_weights, prev_mst_edges):
        current_weights = self.compute_modified_weights()
        weight_changes = current_weights - prev_weights

        if np.all(np.abs(weight_changes) < 1e-6):  # Changed from 1e-4
            mst_cost = sum(current_weights[self.edge_indices[(min(u, v), max(u, v))]] 
                        for u, v in prev_mst_edges)
            mst_length = sum(self.edge_lengths[self.edge_indices[(min(u, v), max(u, v))]] 
                            for u, v in prev_mst_edges)
            if self.verbose:
                print(f"Incremental MST: Reusing previous MST with length={mst_length:.2f}")
            return mst_cost, mst_length, prev_mst_edges

        if self.verbose:
            print(f"Incremental MST: Computing new MST due to weight changes")
        return self.incremental_kruskal(prev_weights, prev_mst_edges, current_weights)

    def solve(self, inherited_cuts=None, inherited_multipliers=None):
        start_time = time()
        
        if inherited_cuts is not None:
            self.best_cuts = inherited_cuts
            self.best_cut_multipliers = inherited_multipliers.copy() if inherited_multipliers else {}
        
        prev_weights = None
        prev_mst_edges = None

        # Bisection variables
        lambda_min = 0.0
        lambda_max = 10.0
        bisection_tolerance = 1e-5

        if self.use_bisection:
            iter_num = 0
            while lambda_max - lambda_min >= bisection_tolerance:
                if prev_weights is not None and prev_mst_edges is not None:
                    mst_cost, mst_length, mst_edges = self.compute_mst_incremental(prev_weights, prev_mst_edges)
                else:
                    mst_cost, mst_length, mst_edges = self.compute_mst()
                
                self.last_mst_edges = mst_edges
                prev_mst_edges = mst_edges
                prev_weights = self.compute_modified_weights()

                is_feasible = mst_length <= self.budget
                self.primal_solutions.append((mst_edges, is_feasible))
                self.multipliers.append(self.lmbda)
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
                        real_weight, _ = self.compute_real_weight_length()
                        if real_weight < self.best_upper_bound:
                            self.best_upper_bound = real_weight
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

                if self.verbose:
                    print(f"Bisection Iteration {iter_num}: lambda={self.lmbda:.6f}, "
                          f"interval={lambda_max - lambda_min:.6f}, subgradient={knapsack_subgradient:.6f}")

                if knapsack_subgradient > 0:
                    lambda_min = self.lmbda
                else:
                    lambda_max = self.lmbda
                self.lmbda = (lambda_min + lambda_max) / 2
                if self.lmbda < 1e-6:
                    self.lmbda = 0.1

                for cut_idx, violation in enumerate(cut_subgradients):
                    current_mult = self.best_cut_multipliers.get(cut_idx, 0)
                    new_mult = max(1e-4, current_mult + self.step_size * violation)
                    if new_mult < 1e-6 and abs(violation) > 1.0:
                        new_mult = 0.1
                    self.best_cut_multipliers[cut_idx] = min(new_mult, 10.0)

                iter_num += 1

            if self.verbose:
                print(f"Bisection converged: interval={lambda_max - lambda_min:.6f} < tolerance={bisection_tolerance}")
        else:
            for iter_num in range(self.max_iter):
                if prev_weights is not None and prev_mst_edges is not None:
                    mst_cost, mst_length, mst_edges = self.compute_mst_incremental(prev_weights, prev_mst_edges)
                else:
                    mst_cost, mst_length, mst_edges = self.compute_mst()
                
                self.last_mst_edges = mst_edges
                prev_mst_edges = mst_edges
                prev_weights = self.compute_modified_weights()

                is_feasible = mst_length <= self.budget
                self.primal_solutions.append((mst_edges, is_feasible))
                self.multipliers.append(self.lmbda)
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
                        real_weight, _ = self.compute_real_weight_length()
                        if real_weight < self.best_upper_bound:
                            self.best_upper_bound = real_weight
                            if self.use_cover_cuts:
                                new_cuts = self.generate_cover_cuts(mst_edges)
                                for cut in new_cuts:
                                    if not any(cut == existing for existing in self.best_cuts):
                                        cut_idx = len(self.best_cuts)
                                        self.best_cuts.append(cut)
                                        self.best_cut_multipliers[cut_idx] = 1.0

                knapsack_subgradient = -(self.budget - mst_length)
                if self.verbose:
                    print(f"Iter {iter_num}: lambda={self.lmbda:.6f}, mst_length={mst_length:.2f}, "
                        f"subgradient={knapsack_subgradient:.2f}, step_size={self.step_size:.6f}")

                # Check for repeated subgradients
                if iter_num > 0 and abs(knapsack_subgradient - self.step_sizes[-2]) < 1e-6:
                    self.consecutive_same_subgradient = getattr(self, 'consecutive_same_subgradient', 0) + 1
                    if self.consecutive_same_subgradient > 10:
                        if self.verbose:
                            print(f"Terminating early: Subgradient {knapsack_subgradient:.2f} repeated {self.consecutive_same_subgradient} times")
                        break
                    # Reset step_size to encourage exploration
                    self.step_size = 0.01
                else:
                    self.consecutive_same_subgradient = 0

                cut_subgradients = [
                    sum(1 for e in mst_edges if e in cut or (e[1], e[0]) in cut) - (len(cut) - 1)
                    for cut in self.best_cuts
                ]
                if self.verbose and cut_subgradients:
                    print(f"Cut subgradients: {cut_subgradients}")

                converged = (abs(knapsack_subgradient) < 1e-5 and 
                            all(abs(g) < 1e-5 for g in cut_subgradients))
                duality_gap = self.best_upper_bound - self.best_lower_bound
                
                if converged or abs(duality_gap) < 1e-5:
                    if self.verbose:
                        print(f"Converged! (Reason: {'Converged' if converged else 'Small duality gap'})")
                    break

                # Adaptive step size (optional alternative)
                # self.step_size = 0.01 / (1 + iter_num / 10)
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
                    if self.verbose:
                        print(f"Cut {cut_idx}: violation={violation:.2f}, multiplier={current_mult:.4f} -> {new_mult:.4f}")

        end_time = time()
        LagrangianMST.total_compute_time += end_time - start_time

        new_cuts = []
        if self.use_cover_cuts and self.best_mst_edges:
            new_cuts = self.generate_cover_cuts(self.best_mst_edges)
            new_cuts = [cut for cut in new_cuts if not any(cut == existing for existing in self.best_cuts)]
        if self.verbose:
            print(f"Final lower bound: {self.best_lower_bound}")
            print(f"Final upper bound: {self.best_upper_bound}")
        return self.best_lower_bound, self.best_upper_bound, new_cuts

    def compute_mst_for_lambda(self, lambda_val):
        modified_edges = []
        for i, (u, v) in enumerate(self.edge_list):
            modified_w = self.edge_weights[i] + lambda_val * self.edge_lengths[i]
            for cut_idx, cut in enumerate(self.best_cuts):
                if (u, v) in cut or (v, u) in cut:
                    modified_w += self.best_cut_multipliers.get(cut_idx, 0)
            modified_edges.append((u, v, modified_w))
        return self.compute_mst(modified_edges)

    def _log_fractional_solution(self, method, edge_weights, msts, elapsed_time):
        if self.verbose:
            total_weight = sum(self.edge_weights[self.edge_indices[e]] * w for e, w in edge_weights.items())
            total_length = sum(self.edge_lengths[self.edge_indices[e]] * w for e, w in edge_weights.items())
            print(f"{method} solution: {len(edge_weights)} edges, "
                  f"weight={total_weight:.2f}, length={total_length:.2f}, time={elapsed_time:.2f}s")
            print(f"MSTs used: {len(msts)}")

    def compute_dantzig_wolfe_solution(self, node):
        """
        Compute a fractional solution using Dantzig-Wolfe decomposition with diverse MSTs.
        Args:
            node: MSTNode object containing fixed_edges, excluded_edges, active_cuts, and cut_multipliers.
        Returns:
            dict: Edge weights {(u, v): weight} where weights are fractional, or None if infeasible.
        """
        start_time = time()
        if len(self.primal_solutions) < 2 or len(self.multipliers) != len(self.primal_solutions):
            if self.verbose:
                print("Insufficient primal solutions for Dantzig-Wolfe")
            return None

        # Collect valid MSTs respecting branching constraints
        valid_msts = []
        for mst_edges, _ in self.primal_solutions:
            if (all((u, v) in mst_edges or (v, u) in mst_edges for u, v in node.fixed_edges) and
                not any((u, v) in mst_edges or (v, u) in mst_edges for u, v in node.excluded_edges)):
                valid_msts.append(mst_edges)
        if len(valid_msts) < 2:
            if self.verbose:
                print(f"Only {len(valid_msts)} valid MSTs after filtering")
            return None

        # Select diverse MSTs (maximize unique edges)
        max_msts = min(10, len(valid_msts))
        selected_msts = []
        covered_edges = set()
        remaining_msts = valid_msts.copy()
        while remaining_msts and len(selected_msts) < max_msts:
            best_mst = None
            best_score = -1
            for mst in remaining_msts:
                new_edges = set((min(u, v), max(u, v)) for u, v in mst) - covered_edges
                score = len(new_edges)
                if score > best_score:
                    best_score = score
                    best_mst = mst
            if best_mst:
                selected_msts.append(best_mst)
                covered_edges.update((min(u, v), max(u, v)) for u, v in best_mst)
                remaining_msts.remove(best_mst)
            else:
                break

        if len(selected_msts) < 2:
            if self.verbose:
                print(f"Only {len(selected_msts)} diverse MSTs selected")
            return None

        if self.verbose:
            print(f"Using {len(selected_msts)} diverse MSTs for Dantzig-Wolfe")

        # Set up LP
        num_msts = len(selected_msts)
        num_edges = len(self.edge_list)
        edge_indices = {(min(u, v), max(u, v)): i for i, (u, v) in enumerate(self.edge_list)}

        # Objective: minimize weighted sum of edge weights + regularization
        c = []
        for k, mst_edges in enumerate(selected_msts):
            weight = sum(self.edge_weights[edge_indices[(min(u, v), max(u, v))]] for u, v in mst_edges)
            c.append(weight + 0.1 * (1.0 / num_msts))

        # Constraints
        A_eq = [np.ones(num_msts)]
        b_eq = [1.0]

        lengths = [sum(self.edge_lengths[edge_indices[(min(u, v), max(u, v))]] for u, v in mst_edges)
                  for mst_edges in selected_msts]
        A_eq.append(lengths)
        b_eq.append(self.budget)

        A_ub = []
        b_ub = []
        for cut in node.active_cuts:
            cut_indices = [edge_indices[(min(u, v), max(u, v))] for u, v in cut if (min(u, v), max(u, v)) in edge_indices]
            if cut_indices:
                row = np.zeros(num_msts)
                for k, mst_edges in enumerate(selected_msts):
                    cut_count = sum(1 for u, v in mst_edges if (min(u, v), max(u, v)) in [self.edge_list[i] for i in cut_indices])
                    row[k] = cut_count
                A_ub.append(row)
                b_ub.append(len(cut) - 1)

        bounds = [(0, None) for _ in range(num_msts)]

        # Solve LP
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            if not res.success:
                if self.verbose:
                    print(f"LP failed: {res.message}")
                return None
            lambda_k = res.x
        except Exception as e:
            if self.verbose:
                print(f"LP solver error: {e}")
            return None

        # Compute edge weights
        edge_weights = {}
        edge_counts = defaultdict(int)
        for u, v in self.edge_list:
            e = (min(u, v), max(u, v))
            weight = sum(lambda_k[k] for k, mst_edges in enumerate(selected_msts)
                        if (u, v) in mst_edges or (v, u) in mst_edges)
            if weight > 1e-6:
                edge_weights[e] = weight
                edge_counts[weight] += 1

        # Log solution
        self._log_fractional_solution("Dantzig-Wolfe", edge_weights, selected_msts, time() - start_time)

        # Debug output
        if self.verbose:
            print(f"Dantzig-Wolfe solution: {len(edge_weights)} edges, total length={sum(self.edge_lengths[edge_indices[e]] * w for e, w in edge_weights.items()):.2f}")
            print(f"LP weights (lambda_k): {lambda_k}")
            print(f"Edge weight distribution: {dict(sorted(edge_counts.items()))}")
            for cut in node.active_cuts:
                cut_count = sum(edge_weights.get((min(u, v), max(u, v)), 0) for u, v in cut)
                print(f"Cut {cut}: count={cut_count:.2f}, rhs={len(cut)-1}")

        # Fallback: average edge frequencies
        unique_weights = len(set(edge_weights.values()))
        if unique_weights < 0.5 * len(edge_weights) and len(valid_msts) > len(selected_msts):
            if self.verbose:
                print("Too many identical weights; falling back to frequency-based weights")
            edge_freq = defaultdict(float)
            for mst in valid_msts:
                for u, v in mst:
                    e = (min(u, v), max(u, v))
                    edge_freq[e] += 1.0 / len(valid_msts)
            edge_weights = {e: w for e, w in edge_freq.items() if w > 1e-6}
            self._log_fractional_solution("Dantzig-Wolfe-Fallback", edge_weights, valid_msts, time() - start_time)

        return edge_weights if edge_weights else None

    def recover_primal_solution(self, node):
        """
        Recover a feasible primal solution (spanning tree) satisfying budget, cover cuts, and branching constraints.
        Args:
            node: MSTNode object containing fixed_edges, excluded_edges, active_cuts, and cut_multipliers.
        Returns:
            tuple: (mst_edges, real_weight, real_length) where mst_edges is a list of edges,
                   real_weight is the total weight, and real_length is the total length.
                   Returns (None, float('inf'), float('inf')) if no feasible solution is found.
        """
        start_time = time()

        # Step 1: Check stored primal solutions
        for mst_edges, is_feasible in self.primal_solutions:
            # Verify branching constraints
            if not all((u, v) in mst_edges or (v, u) in mst_edges for u, v in node.fixed_edges):
                continue
            if any((u, v) in mst_edges or (v, u) in mst_edges for u, v in node.excluded_edges):
                continue

            # Verify budget constraint
            real_length = sum(self.edge_lengths[self.edge_indices[(min(u, v), max(u, v))]] 
                              for u, v in mst_edges)
            if real_length > self.budget:
                continue

            # Verify cover cuts
            valid_cuts = True
            for cut in node.active_cuts:
                cut_count = sum(1 for u, v in mst_edges if (u, v) in cut or (v, u) in cut)
                if cut_count > len(cut) - 1:
                    valid_cuts = False
                    break
            if not valid_cuts:
                continue

            # Verify spanning tree properties
            uf = UnionFind(self.num_nodes)
            for u, v in mst_edges:
                uf.union(u, v)
            if uf.count_components() != 1 or len(set(u for u, _ in mst_edges) | set(v for _, v in mst_edges)) < self.num_nodes:
                continue

            # Compute real weight
            real_weight = sum(self.edge_weights[self.edge_indices[(min(u, v), max(u, v))]] 
                              for u, v in mst_edges)
            end_time = time()
            if self.verbose:
                print(f"Feasible primal solution found from primal_solutions: weight={real_weight:.2f}, length={real_length:.2f}")
            return mst_edges, real_weight, real_length

        # Step 2: Greedy heuristic (modified Kruskal's)
        uf = UnionFind(self.num_nodes)
        mst_edges = []
        total_length = 0.0
        total_weight = 0.0

        # Add fixed edges first
        for edge_idx in self.fixed_edge_indices:
            u, v = self.edge_list[edge_idx]
            if uf.union(u, v):
                mst_edges.append((u, v))
                total_length += self.edge_lengths[edge_idx]
                total_weight += self.edge_weights[edge_idx]
            else:
                if self.verbose:
                    print(f"Fixed edge ({u}, {v}) creates cycle in greedy heuristic")
                return None, float('inf'), float('inf')

        # Sort edges by weight
        edge_indices = [i for i in range(len(self.edges)) 
                        if i not in self.fixed_edge_indices and i not in self.excluded_edge_indices]
        sorted_edges = sorted(edge_indices, key=lambda i: self.edge_weights[i])

        for edge_idx in sorted_edges:
            u, v = self.edge_list[edge_idx]
            new_length = total_length + self.edge_lengths[edge_idx]
            if new_length > self.budget:
                continue

            # Check cover cuts
            temp_edges = mst_edges + [(u, v)]
            valid_cuts = True
            for cut in node.active_cuts:
                cut_count = sum(1 for x, y in temp_edges if (x, y) in cut or (y, x) in cut)
                if cut_count > len(cut) - 1:
                    valid_cuts = False
                    break
            if not valid_cuts:
                continue

            if uf.union(u, v):
                mst_edges.append((u, v))
                total_length = new_length
                total_weight += self.edge_weights[edge_idx]

        # Verify spanning tree
        if uf.count_components() != 1 or len(set(u for u, _ in mst_edges) | set(v for _, v in mst_edges)) < self.num_nodes:
            if self.verbose:
                print("Greedy heuristic failed to produce a valid spanning tree")
            return None, float('inf'), float('inf')

        end_time = time()
        if self.verbose:
            print(f"Feasible primal solution found via greedy heuristic: weight={total_weight:.2f}, length={total_length:.2f}")
        return mst_edges, total_weight, total_length

    def compute_real_weight_length(self):
        real_weight = sum(self.edge_weights[self.edge_indices[(min(u, v), max(u, v))]] 
                          for u, v in self.last_mst_edges)
        real_length = sum(self.edge_lengths[self.edge_indices[(min(u, v), max(u, v))]] 
                          for u, v in self.last_mst_edges)
        return real_weight, real_length




