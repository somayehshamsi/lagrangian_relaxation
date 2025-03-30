import networkx as nx
import numpy as np
import random
from time import time

class LagrangianMST:

    total_compute_time = 0

    def __init__(self, edges, num_nodes, budget, fixed_edges=None, excluded_edges=None, initial_lambda=0.1, step_size=0.005, max_iter=50, p=0.95):
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
        LagrangianMST.total_compute_time += end_time - start_time


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

    def solve(self):
        """
        Solve the Lagrangian Relaxation problem using subgradient optimization.
        """
        start_time = time()
        for iter_num in range(self.max_iter):
            # Modify edge weights using current lambda
            modified_edges = [(u, v, w + self.lmbda * l) for u, v, w, l in self.edges]
            # print("lamda", self.lmbda)

            # Compute MST with modified costs
            mst_cost, mst_length, mst_edges = self.compute_mst(modified_edges)
            self.last_mst_edges = mst_edges 

                    # Label feasibility (store as tuple: (mst_edges, is_feasible))
            is_feasible = mst_length <= self.budget
            self.primal_solutions.append((mst_edges, is_feasible))
            
            # Store primal solution and step size
            # self.primal_solutions.append(mst_edges)
            self.step_sizes.append(self.step_size) 

            #             # Debugging output
            # print(f"Iteration {iter_num}: Step Size = {self.step_size}, Lambda = {self.lmbda}")
            # print(f"MST Edges: {mst_edges}")
                        
            # Compute Lagrangian lower bound
            lagrangian_bound = mst_cost - self.lmbda * (self.budget)
            # self.best_lower_bound = max(self.best_lower_bound, lagrangian_bound)
            if lagrangian_bound > self.best_lower_bound:
                self.best_lower_bound = lagrangian_bound
                self.best_lambda = self.lmbda  # Update the best lambda value
                self.best_mst_edges = mst_edges

            # Update the best upper bound if MST is feasible
            if mst_length <= self.budget and nx.is_connected(nx.Graph(mst_edges)):
                self.best_upper_bound = min(self.best_upper_bound, mst_cost)

            # Compute the subgradient
            subgradient = -(self.budget - mst_length)

            # Check for convergence
            duality_gap = self.best_upper_bound - self.best_lower_bound
            if abs(subgradient) < 1e-5 and abs(duality_gap) < 1e-5:
                print("Converged!")
                break

            # Adaptive step size update using geometric decay
            self.step_size *= self.p
            # self.lmbda = max(0, self.lmbda + self.step_size * subgradient)
            self.lmbda =  self.lmbda + self.step_size * subgradient
            
        end_time = time()
        LagrangianMST.total_compute_time += end_time - start_time

        return self.best_lower_bound, self.best_upper_bound
    
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
