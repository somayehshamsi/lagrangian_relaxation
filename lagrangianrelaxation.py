import networkx as nx
import numpy as np
import random
from time import time

class LagrangianMST:

    total_compute_time = 0

    def __init__(self, edges, num_nodes, budget, fixed_edges=None, excluded_edges=None, initial_lambda=0.1, step_size=0.005, max_iter=50, p=0.95, use_cover_cuts=False, cut_frequency=5 ):
        start_time = time()
        self.edges = edges
        self.num_nodes = num_nodes
        self.budget = budget
        self.fixed_edges = fixed_edges if fixed_edges is not None else set()
        self.excluded_edges = excluded_edges if excluded_edges is not None else set()
        self.lmbda = initial_lambda
        self.step_size = step_size
        self.p = p  # Geometric decay factor for step size
        self.max_iter = max_iter
        self.best_lower_bound = float('-inf')
        self.best_upper_bound = float('inf')
        self.last_mst_edges = []
        end_time = time()
        self.primal_solutions = []  # Store primal solutions (MSTs)
        self.step_sizes = []  # Store step sizes (λₖ)
        self.best_lambda = initial_lambda
        self.best_mst_edges = None  # Track the MST edges that give the best lower bound
        self.best_cost = 0

        self.use_cover_cuts = use_cover_cuts
        self.cut_frequency = cut_frequency
        self.cover_cuts = []
        self.cover_cut_multipliers = {}

        # #newww 3 line
        # self.edge_length = {(u,v): l for u,v,w,l in edges}  # Cache lengths
        # self.trust_region = 1.0  # For multiplier stabilization
        # self.cut_pool = []       # Global cut storage across nodes

        LagrangianMST.total_compute_time += end_time - start_time


    def generate_cover_cuts(self, mst_edges):
        if not mst_edges:
            return []

        # Calculate total length of current MST
        total_length = sum(next(l for x,y,w,l in self.edges 
                            if (x,y)==(u,v) or (y,x)==(u,v)) 
                        for u,v in mst_edges)
        
        # Only generate cuts if current solution violates budget
        if total_length <= self.budget:
            return []
        
            # Filter edges to only consider active edges (not fixed or excluded)
        active_edges = [(u,v) for u,v in mst_edges 
                   if (u,v) not in self.excluded_edges and 
                      (v,u) not in self.excluded_edges and
                      (u,v) not in self.fixed_edges and 
                      (v,u) not in self.fixed_edges]

        # 1. Identify the most problematic edges (main contributors to budget violation)
        edge_contributions = []
        for u, v in active_edges:
            l = next(l for x,y,w,l in self.edges 
                    if (x,y)==(u,v) or (y,x)==(u,v))
            w = next(w for x,y,w,l in self.edges 
                    if (x,y)==(u,v) or (y,x)==(u,v))
            # Score based on both length and weight-to-length ratio
            score = l * (1 + (w/l if l > 0 else 0))
            edge_contributions.append(((u,v), l, w, score))
        
        # Sort edges by their contribution score (descending)
        edge_contributions.sort(key=lambda x: -x[3])

        cuts = []
        
        # 2. Generate cuts based on edge combinations
        # Strategy 1: Pairs of high-contributing edges
        for i in range(min(5, len(edge_contributions))):
            for j in range(i+1, min(i+4, len(edge_contributions))):
                e1, l1, w1, _ = edge_contributions[i]
                e2, l2, w2, _ = edge_contributions[j]
                if l1 + l2 > self.budget:
                    cuts.append({e1, e2})
        
        # Strategy 2: Critical triplets
        for i in range(min(3, len(edge_contributions))):
            for j in range(i+1, min(i+3, len(edge_contributions))):
                for k in range(j+1, min(j+3, len(edge_contributions))):
                    e1, l1, w1, _ = edge_contributions[i]
                    e2, l2, w2, _ = edge_contributions[j]
                    e3, l3, w3, _ = edge_contributions[k]
                    if l1 + l2 + l3 > self.budget:
                        cuts.append({e1, e2, e3})
        
        # Strategy 3: Path-based cuts (identify long paths in the MST)
        if len(mst_edges) >= 3:
            G = nx.Graph(mst_edges)
            degrees = dict(G.degree())
            # Find high-degree nodes as potential cut points
            hub_nodes = [n for n in degrees if degrees[n] > 2]
            for node in hub_nodes[:3]:  # Limit to top 3 hubs
                neighbors = list(G.neighbors(node))
                # Create cuts from hub-spoke combinations
                for i in range(len(neighbors)):
                    for j in range(i+1, len(neighbors)):
                        e1 = (node, neighbors[i])
                        e2 = (node, neighbors[j])
                        l1 = next(l for x,y,w,l in self.edges 
                                if (x,y)==e1 or (y,x)==e1)
                        l2 = next(l for x,y,w,l in self.edges 
                                if (x,y)==e2 or (y,x)==e2)
                        if l1 + l2 > self.budget:
                            cuts.append({e1, e2})
        
        # 3. Add the most obvious cut - the entire MST if it violates budget
        if total_length > self.budget and len(mst_edges) >= 2:
            cuts.append(set(mst_edges))
        
        # Remove duplicate cuts
        unique_cuts = []
        seen = set()
        for cut in cuts:
            frozen = frozenset(cut)
            if frozen not in seen:
                seen.add(frozen)
                unique_cuts.append(cut)
        
        return unique_cuts[:10]  # Return at most 10 strongest cuts


    
    def compute_mst(self, modified_edges):
        # Create a graph with the modified edges
        G = nx.Graph()
        G.add_weighted_edges_from(modified_edges)

        # Ensure fixed edges are included in the graph
        for u, v in self.fixed_edges:
            # Find the corresponding edge in the modified edges
            for edge in modified_edges:
                if (edge[0], edge[1]) == (u, v) or (edge[0], edge[1]) == (v, u):
                    G.add_edge(u, v, weight=edge[2])
                    break

        # Exclude edges that are in excluded_edges
        for u, v in self.excluded_edges:
            if G.has_edge(u, v):
                G.remove_edge(u, v)

        # If there are fixed edges, check if they form a cycle
        if self.fixed_edges:
            fixed_graph = nx.Graph()
            fixed_graph.add_edges_from(self.fixed_edges)
            if not nx.is_forest(fixed_graph):  # Check if the fixed edges form a cycle
                return float('inf'), float('inf'), []  # Prune this node (infeasible)

        all_edges = sorted(G.edges(data=True), key=lambda x: x[2]['weight'])

        # Initialize the MST with the fixed edges
        mst_edges = list(self.fixed_edges)
        mst_graph = nx.Graph()
        mst_graph.add_edges_from(self.fixed_edges)

        # Use a data structure to detect cycles
        parent = {node: node for node in G.nodes()}

        def find(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]  
                u = parent[u]
            return u

        def union(u, v):
            u_root = find(u)
            v_root = find(v)
            if u_root == v_root:
                return False  # Cycle detected
            parent[v_root] = u_root
            return True

        # Add the fixed edges to the Union-Find structure
        for u, v in self.fixed_edges:
            union(u, v)

        # Add the remaining edges to the MST
        for u, v, data in all_edges:
            if (u, v) not in self.fixed_edges and (v, u) not in self.fixed_edges:
                if union(u, v):
                    mst_edges.append((u, v))
                    mst_graph.add_edge(u, v, weight=data['weight'])

        # Check if the MST is connected and includes all nodes
        if not nx.is_connected(mst_graph) or len(mst_graph.nodes) < self.num_nodes:
            return float('inf'), float('inf'), []  # Prune this node (infeasible)

        # Calculate the total cost and length of the MST
        mst_cost = sum(G[u][v]['weight'] for u, v in mst_edges)
        mst_length = sum(next(l for x, y, w, l in self.edges if (x, y) == (u, v) or (y, x) == (u, v)) for u, v in mst_edges)

        return mst_cost, mst_length, mst_edges


    def solve(self, inherited_cuts=None, inherited_multipliers=None):
        """
        Solve the Lagrangian Relaxation problem using subgradient optimization.
        inherited_cuts: Cuts from parent node that should be relaxed in this node
        inherited_multipliers: Corresponding multipliers for inherited cuts
        Returns: (best_lower_bound, best_upper_bound, new_cuts)
        """
        start_time = time()
        
        # Initialize with inherited cuts if provided
        if inherited_cuts is not None:
            self.cover_cuts = inherited_cuts
            self.cover_cut_multipliers = inherited_multipliers.copy() if inherited_multipliers else {}
        
        for iter_num in range(self.max_iter):
            # Generate cover cuts from best integer solution (if enabled)
            # Note: These new cuts won't be relaxed in current node, only passed to children
            if (self.use_cover_cuts and 
                self.best_mst_edges and 
                iter_num % self.cut_frequency == 0):
                
                new_cuts = self.generate_cover_cuts(self.best_mst_edges)
                for cut in new_cuts:
                    if not any(cut == existing for existing in self.cover_cuts):
                        cut_idx = len(self.cover_cuts)
                        self.cover_cuts.append(cut)
                        self.cover_cut_multipliers[cut_idx] = 1.0  # Initialize new cut multiplier

            # Modify edge weights with lambda and cuts
            modified_edges = []
            for u, v, w, l in self.edges:
                modified_w = w + self.lmbda * l
                for cut_idx, cut in enumerate(self.cover_cuts):
                    if (u, v) in cut or (v, u) in cut:
                        modified_w += self.cover_cut_multipliers.get(cut_idx, 0)
                modified_edges.append((u, v, modified_w))

            # Compute MST with modified costs
            mst_cost, mst_length, mst_edges = self.compute_mst(modified_edges)
            self.last_mst_edges = mst_edges

            # Store primal solution and feasibility
            is_feasible = mst_length <= self.budget
            self.primal_solutions.append((mst_edges, is_feasible))
            self.step_sizes.append(self.step_size)

            # Calculate cover cut penalty terms
            cover_cut_penalty = sum(
                multiplier * (len(cut) - 1)
                for cut_idx, cut in enumerate(self.cover_cuts)
                for multiplier in [self.cover_cut_multipliers.get(cut_idx, 0)]
            )

            # Compute Lagrangian bound
            lagrangian_bound = mst_cost - self.lmbda * self.budget - cover_cut_penalty

            # Update best lower bound
            if lagrangian_bound > self.best_lower_bound:
                self.best_lower_bound = lagrangian_bound
                self.best_lambda = self.lmbda
                self.best_mst_edges = mst_edges
                self.best_cost = mst_cost

            # Update best upper bound if feasible
            if is_feasible and nx.is_connected(nx.Graph(mst_edges)):
                if mst_cost < self.best_upper_bound:
                    self.best_upper_bound = mst_cost
                    # When we find a better feasible solution, we can generate additional cuts
                    if self.use_cover_cuts:
                        new_cuts = self.generate_cover_cuts(mst_edges)
                        for cut in new_cuts:
                            if not any(cut == existing for existing in self.cover_cuts):
                                cut_idx = len(self.cover_cuts)
                                self.cover_cuts.append(cut)
                                self.cover_cut_multipliers[cut_idx] = 1.0

            # Compute subgradients
            # 1. Main knapsack constraint subgradient
            knapsack_subgradient = -(self.budget - mst_length)
            
            # 2. Cover cut subgradients (vector of violations)
            cut_subgradients = []
            for cut_idx, cut in enumerate(self.cover_cuts):
                # Count how many edges from this cut are in the MST
                edges_in_mst = sum(1 for e in mst_edges if e in cut or (e[1], e[0]) in cut)
                violation = edges_in_mst - (len(cut) - 1)
                cut_subgradients.append(violation)
            
            # Check convergence for all constraints
            converged = (abs(knapsack_subgradient) < 1e-5 and 
                        all(abs(g) < 1e-5 for g in cut_subgradients))
            duality_gap = self.best_upper_bound - self.best_lower_bound
            
            if converged or abs(duality_gap) < 1e-5:
                print("Converged!")
                break

            # Adaptive step size update
            self.step_size *= self.p  # Geometric decay


            # Update main lambda (knapsack constraint)
            self.lmbda = max(0, self.lmbda + self.step_size * knapsack_subgradient)
            
            # Update cut multipliers (vector of multipliers)
            for cut_idx, violation in enumerate(cut_subgradients):
                current_mult = self.cover_cut_multipliers.get(cut_idx, 0)
                new_mult = max(0, current_mult + self.step_size * violation)
                self.cover_cut_multipliers[cut_idx] = min(new_mult, 10.0)  # Cap at 10.0
            

        end_time = time()
        LagrangianMST.total_compute_time += end_time - start_time
        print("transfered_cut", self.cover_cuts)

        # # Generate new cuts from best solution (not relaxed in this node)
        new_cuts = []
        if self.use_cover_cuts and self.best_mst_edges:
            new_cuts = self.generate_cover_cuts(self.best_mst_edges)
            # Filter out cuts that are already active
            new_cuts = [cut for cut in new_cuts if not any(cut == existing for existing in self.cover_cuts)]
        


        return self.best_lower_bound, self.best_upper_bound, new_cuts
    
    def compute_shor_primal_solution(self):
        """
        Compute the Shor primal solution as a weighted average of all MSTs found during optimization.
        Returns a dictionary mapping edges to their average weights.
        """
        if not self.primal_solutions:
            return None
            
        # Initialize edge weights
        edge_weights = {}
        total_weight = 0.0
        
        # Sum up all weights
        for i, mst_edges in enumerate(self.primal_solutions):
            weight = self.step_sizes[i]  # Use step size as weight (λₖ)
            total_weight += weight
            for edge in mst_edges:
                if edge in edge_weights:
                    edge_weights[edge] += weight
                else:
                    edge_weights[edge] = weight
        
        # Normalize by total weight
        if total_weight > 0:
            for edge in edge_weights:
                edge_weights[edge] /= total_weight
        
        return edge_weights


    def compute_dantzig_wolfe_solution(self):
        """Compute the LP-optimal solution via convex combination of two MSTs"""
        if len(self.primal_solutions) < 2:
            return None

        # Get most recent feasible/infeasible pair
        feasible = [mst for mst, feasible in self.primal_solutions if feasible]
        infeasible = [mst for mst, feasible in self.primal_solutions if not feasible]
        
        if not feasible or not infeasible:
            return None

        mst_feas = feasible[-1]
        mst_infeas = infeasible[-1]

        # Calculate total lengths
        def get_length(mst):
            return sum(l for u,v,w,l in self.edges 
                    if (u,v) in mst or (v,u) in mst)

        len_feas = get_length(mst_feas)
        len_infeas = get_length(mst_infeas)

        # Compute convex combination weight α
        try:
            α = (self.budget - len_feas) / (len_infeas - len_feas)
            α = max(0, min(1, α))  # Clamp to [0,1]
        except ZeroDivisionError:
            return None

        # Build combined solution
        combined = {}
        all_edges = set(mst_feas) | set(mst_infeas)
        
        for e in all_edges:
            in_feas = e in mst_feas or (e[1],e[0]) in mst_feas
            in_infeas = e in mst_infeas or (e[1],e[0]) in mst_infeas
            
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
        """
        Compute the real total weight and length of the last MST found.
        """
        real_weight = sum(next(w for x, y, w, l in self.edges if (x, y) == (u, v) or (y, x) == (u, v)) for u, v in self.last_mst_edges)
        real_length = sum(next(l for x, y, w, l in self.edges if (x, y) == (u, v) or (y, x) == (u, v)) for u, v in self.last_mst_edges)

        return real_weight, real_length
