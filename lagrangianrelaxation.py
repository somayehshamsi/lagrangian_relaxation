import argparse
import networkx as nx
import numpy as np
import random

# we create the parser for the program
parser = argparse.ArgumentParser(prog='MST Lagrangean B&B', usage='%(prog)s [options]')
parser.add_argument("rule")
args = parser.parse_args()
print(args.rule)

class LagrangianMST:
    def __init__(self, edges, num_nodes, budget, initial_lambda=1.0, step_size=1.0, max_iter=10, p=0.95):

        self.edges = edges
        self.num_nodes = num_nodes
        self.budget = budget
        self.lmbda = initial_lambda
        self.step_size = step_size
        self.p = p  # Geometric decay factor for step size
        self.max_iter = max_iter
        self.best_lower_bound = float('-inf')
        self.best_upper_bound = float('inf')
        self.last_mst_edges = []


    def compute_mst(self, modified_edges):

        G = nx.Graph()
        G.add_weighted_edges_from(modified_edges)
        mst = nx.minimum_spanning_tree(G, weight='weight')

        mst_cost = sum(G[u][v]['weight'] for u, v in mst.edges)
        mst_length = sum(next(l for x, y, w, l in self.edges if (x, y) == (u, v) or (y, x) == (u, v)) for u, v in mst.edges)

        return mst_cost, mst_length, list(mst.edges)

    def solve(self):
        """
        Solve the Lagrangian Relaxation problem using subgradient optimization.
        """
        for iter_num in range(self.max_iter):
            # Modify edge weights using current lambda
            modified_edges = [(u, v, w + self.lmbda * l) for u, v, w, l in self.edges]

            # Compute MST with modified costs
            mst_cost, mst_length, mst_edges = self.compute_mst(modified_edges)
            self.last_mst_edges = mst_edges  


            # Compute Lagrangian lower bound
            lagrangian_bound = mst_cost - self.lmbda * (self.budget - mst_length)
            self.best_lower_bound = max(self.best_lower_bound, lagrangian_bound)

            # Print debug info
            print(f"Iteration {iter_num}:")
            print(f"  λ = {self.lmbda:.4f}")
            print(f"  MST Cost (Modified Weights): {mst_cost}")
            print(f"  MST Length: {mst_length}")
            print(f"  Lagrangian Lower Bound: {lagrangian_bound}")
            print(f"  Budget Constraint Violation: {self.budget - mst_length}")
            print(f"  Edges in MST: {mst_edges}")

            # Update the best upper bound if MST is feasible
            if mst_length <= self.budget:
                self.best_upper_bound = min(self.best_upper_bound, mst_cost)

            # Compute the subgradient
            subgradient = -(self.budget - mst_length)  # 

            # Check for convergence
            duality_gap = self.best_upper_bound - self.best_lower_bound
            if abs(subgradient) < 1e-5 or duality_gap < 1e-3:
                print("Converged!")
                break

            # Adaptive step size update using geometric decay
            self.step_size *= self.p
            self.lmbda = max(0, self.lmbda + self.step_size * subgradient)

        return self.best_lower_bound, self.best_upper_bound
    

    def compute_real_weight_length(self):
        """
        Compute the real total weight and length of the last MST found.
        """
        real_weight = sum(next(w for x, y, w, l in self.edges if (x, y) == (u, v) or (y, x) == (u, v)) for u, v in self.last_mst_edges)
        real_length = sum(next(l for x, y, w, l in self.edges if (x, y) == (u, v) or (y, x) == (u, v)) for u, v in self.last_mst_edges)

        return real_weight, real_length

# # Generate a complex graph with 15 nodes and 30 edges
# random.seed(42)  # For reproducibility
# num_nodes = 15
# edges = []

# # Generate 30 random edges with weights (costs) between 5 and 50, and lengths between 1 and 10
# for _ in range(30):
#     u, v = random.sample(range(num_nodes), 2)
#     weight = random.randint(5, 50)
#     length = random.randint(1, 10)
#     edges.append((u, v, weight, length))

# budget = 50  # Maximum allowed total length
edges = [
    (0, 1, 10, 3), (0, 2, 15, 4), (0, 3, 20, 5), (1, 4, 25, 6), (1, 5, 30, 3),
    (2, 6, 12, 2), (2, 7, 18, 4), (3, 8, 22, 5), (3, 9, 28, 7), (4, 10, 35, 8),
    (4, 11, 40, 9), (5, 12, 38, 7), (5, 13, 45, 5), (6, 14, 20, 6), (7, 8, 14, 3),
    (7, 9, 26, 5), (8, 10, 19, 4), (9, 11, 33, 6), (10, 12, 50, 9), (11, 13, 27, 7),
    (12, 14, 32, 6), (13, 14, 24, 5), (1, 6, 18, 4), (2, 5, 17, 3), (3, 7, 21, 5),
    (4, 8, 29, 7), (5, 9, 36, 6), (6, 10, 23, 5), (7, 11, 31, 6)
]

num_nodes = 15
budget = 60  # Maximum allowed MST length


mst_solver = LagrangianMST(edges, num_nodes, budget)
lower_bound, upper_bound = mst_solver.solve()
real_weight, real_length = mst_solver.compute_real_weight_length()


print(f"\nFinal Lagrangian Lower Bound: {lower_bound}")
print(f"Final Best Upper Bound (Feasible MST within budget): {upper_bound}")
print(f"Real Weight of Last MST: {real_weight}, Real Length of Last MST: {real_length}")


