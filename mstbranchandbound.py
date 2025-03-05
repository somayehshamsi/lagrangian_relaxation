# """
# Branch-and-bound implementation for the MST problem.
# """
# import branchandbound

# class MSTNode(branchandbound.Node):
#     """
    
#     """
    
#     def create_children(self, edge: object):
#         """
#         Creates and returns two children nodes that partition
#         the space of solutions according to the given branching object.
#         """
#         pass
import heapq
import networkx as nx
from lagrangianrelaxation import LagrangianMST
from branchandbound import Node, BranchAndBound, RandomBranchingRule

class MSTNode(Node):
    def __init__(self, edges, num_nodes, budget, fixed_edges=set(), excluded_edges=set(), branched_edges=set()):
        self.edges = edges
        self.num_nodes = num_nodes
        self.budget = budget
        self.fixed_edges = set(fixed_edges)  # Edges forced to be in the MST
        self.excluded_edges = set(excluded_edges)  # Edges forced to be excluded from the MST
        self.branched_edges = set(branched_edges)  # Edges that have already been branched on

        # Filter edges: Exclude edges that are in excluded_edges
        filtered_edges = [(u, v, w, l) for (u, v, w, l) in edges if (u, v) not in self.excluded_edges]

        # Solve Lagrangian relaxation
        self.lagrangian_solver = LagrangianMST(filtered_edges, num_nodes, budget, self.fixed_edges, self.excluded_edges)
        self.local_lower_bound, _ = self.lagrangian_solver.solve()
        self.actual_cost, _ = self.lagrangian_solver.compute_real_weight_length()
        self.local_lower_bound = self.actual_cost

        # Store the MST edges from the Lagrangian solver
        self.mst_edges = self.lagrangian_solver.last_mst_edges

        # Initialize the Node superclass
        super().__init__(self.local_lower_bound)

    def __lt__(self, other):
        return self.local_lower_bound < other.local_lower_bound

    def create_children(self, branched_edge):
        u, v = branched_edge[0], branched_edge[1]  # Extract (u, v) only

        # Add the branched edge to the set of branched edges
        new_branched_edges = self.branched_edges | {(u, v)}

        # Create child nodes
        fixed_child = MSTNode(self.edges, self.num_nodes, self.budget,
                            self.fixed_edges | {(u, v)}, self.excluded_edges, new_branched_edges)
        excluded_child = MSTNode(self.edges, self.num_nodes, self.budget,
                                self.fixed_edges, self.excluded_edges | {(u, v)}, new_branched_edges)
        return [fixed_child, excluded_child]

    def is_feasible(self):
        # Compute the real weight and length of the MST
        real_weight, real_length = self.lagrangian_solver.compute_real_weight_length()

        # Check if the MST length is within the budget
        if real_length > self.budget:
            return False, "MST length exceeds budget"

        # Check if the MST includes all nodes
        mst_nodes = set()
        for u, v in self.mst_edges:
            mst_nodes.add(u)
            mst_nodes.add(v)

        if len(mst_nodes) < self.num_nodes:
            return False, "MST does not include all nodes"

        # Check if the MST is connected
        mst_graph = nx.Graph(self.mst_edges)
        if not nx.is_connected(mst_graph):
            return False, "MST is not connected"

        return True, "MST is feasible"

    def compute_upper_bound(self):
        real_weight, _ = self.lagrangian_solver.compute_real_weight_length()
        return real_weight

    def get_branching_candidates(self):
        # Exclude edges that have already been branched on
        assert self.mst_edges
        candidate_edges = [e for e in self.mst_edges if (e[0], e[1]) not in self.fixed_edges and
                          (e[0], e[1]) not in self.excluded_edges and
                          (e[0], e[1]) not in self.branched_edges]
        return candidate_edges if candidate_edges else None


if __name__ == "__main__":
    edges = [
        (0, 1, 10, 3), (0, 2, 15, 4), (0, 3, 20, 5), (1, 4, 25, 6), (1, 5, 30, 3),
        (2, 6, 12, 2), (2, 7, 18, 4), (3, 8, 22, 5), (3, 9, 28, 7), (4, 10, 35, 8),
        (4, 11, 40, 9), (5, 12, 38, 7), (5, 13, 45, 5), (6, 14, 20, 6), (7, 8, 14, 3),
        (7, 9, 26, 5), (8, 10, 19, 4), (9, 11, 33, 6), (10, 12, 50, 9), (11, 13, 27, 7),
        (12, 14, 32, 6), (13, 14, 24, 5), (1, 6, 18, 4), (2, 5, 17, 3), (3, 7, 21, 5)
        # ,
        # (4, 8, 29, 7), (5, 9, 36, 6), (6, 10, 23, 5), (7, 11, 31, 6)
    ]
    num_nodes = 15
    budget = 60
    # edges = [
    #     (0, 1, 10, 3), (0, 2, 15, 4), (0, 3, 20, 5), (1, 2, 12, 2), (1, 3, 18, 4),
    #     (1, 4, 25, 6), (2, 3, 14, 3), (2, 5, 17, 3), (3, 4, 22, 5), (3, 5, 28, 7),
    #     (3, 6, 30, 8), (4, 5, 35, 9), (4, 7, 40, 10), (5, 6, 38, 7), (5, 7, 45, 5),
    #     (5, 8, 50, 9), (6, 7, 27, 6), (6, 8, 33, 7), (6, 9, 36, 8), (7, 8, 29, 5)
    # ]
    # num_nodes = 10  # Number of nodes in the graph
    # budget = 40     # Maximum allowed MST length


    root_node = MSTNode(edges, num_nodes, budget)
    branching_rule = RandomBranchingRule()
    bnb_solver = BranchAndBound(branching_rule)
    best_solution, best_upper_bound = bnb_solver.solve(root_node)

    # Print the optimal MST cost and edges
    print(f"Optimal MST Cost within Budget: {best_upper_bound}")
    if best_solution:
        print("Edges in the Optimal MST:")
        for edge in best_solution.mst_edges:
            print(edge)
    else:
        print("No feasible solution found.")

    print(f"Lagrangian MST time: {LagrangianMST.total_compute_time:.2f}s")