import heapq
import random
import networkx as nx
from lagrangianrelaxation import LagrangianMST
from branchandbound import Node, BranchAndBound, RandomBranchingRule

class MSTNode(Node):
    def __init__(self, edges, num_nodes, budget, fixed_edges=set(), excluded_edges=set(), branched_edges=set(), initial_lambda = 1.0, inherit_lambda = False, branching_rule="random_mst",
                 step_size=1.0, inherit_step_size=False):
        self.edges = edges
        self.num_nodes = num_nodes
        self.budget = budget
        self.fixed_edges = set(fixed_edges)  # Edges forced to be in the MST
        self.excluded_edges = set(excluded_edges)  # Edges forced to be excluded from the MST
        self.branched_edges = set(branched_edges)  # Edges that have already been branched on

        self.inherit_lambda = inherit_lambda  # Whether to inherit lambda from the parent
        self.initial_lambda = initial_lambda if inherit_lambda else 1.0  # Reset to 1.0 if not inheriting
        self.branching_rule = branching_rule  # Branching rule: random_mst or random_all
        self.step_size = step_size  # Current step size
        self.inherit_step_size = inherit_step_size  # Whether to inherit step size from the parent


        # Filter edges: Exclude edges that are in excluded_edges
        filtered_edges = [(u, v, w, l) for (u, v, w, l) in edges if (u, v) not in self.excluded_edges]

        # Solve Lagrangian relaxation
        self.lagrangian_solver = LagrangianMST(filtered_edges, num_nodes, budget, self.fixed_edges, self.excluded_edges, initial_lambda=self.initial_lambda, step_size=self.step_size)
        self.local_lower_bound, _ = self.lagrangian_solver.solve()
        self.actual_cost, _ = self.lagrangian_solver.compute_real_weight_length()
        # self.local_lower_bound = self.actual_cost

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

        # Get the current lambda value from the Lagrangian solver
        current_lambda = self.lagrangian_solver.lmbda if self.inherit_lambda else 1.0
        current_step_size = self.lagrangian_solver.step_size if self.inherit_step_size else 1.0


        # Create child nodes
        fixed_child = MSTNode(self.edges, self.num_nodes, self.budget,
                            self.fixed_edges | {(u, v)}, self.excluded_edges, new_branched_edges, initial_lambda=current_lambda, inherit_lambda=self.inherit_lambda, branching_rule=self.branching_rule, step_size=current_step_size, inherit_step_size=self.inherit_step_size )
        excluded_child = MSTNode(self.edges, self.num_nodes, self.budget,
                                self.fixed_edges, self.excluded_edges | {(u, v)}, new_branched_edges, initial_lambda=current_lambda, inherit_lambda=self.inherit_lambda, branching_rule=self.branching_rule, step_size=current_step_size, inherit_step_size=self.inherit_step_size)
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

        if self.branching_rule == "most_violated":
            # Branch on the edge with the highest weight-to-length ratio
            candidate_edges = sorted(
                [(u, v, w, l) for u, v, w, l in self.edges if (u, v) not in self.fixed_edges and (u, v) not in self.excluded_edges],
                key=lambda x: x[2] / x[3],  # Weight-to-length ratio
                reverse=True,
            )
        elif self.branching_rule == "random_mst":
            # Branch only from edges in the MST
            assert self.mst_edges
            candidate_edges = [e for e in self.mst_edges if (e[0], e[1]) not in self.fixed_edges and
                              (e[0], e[1]) not in self.excluded_edges and
                              (e[0], e[1]) not in self.branched_edges]
        elif self.branching_rule == "random_all":
            # Branch from all candidate edges (not just MST edges)
            candidate_edges = [(u, v) for (u, v, w, l) in self.edges if (u, v) not in self.fixed_edges and
                              (u, v) not in self.excluded_edges and
                              (u, v) not in self.branched_edges]
        else:
            raise ValueError(f"Unknown branching rule: {self.branching_rule}")
        return candidate_edges if candidate_edges else None
