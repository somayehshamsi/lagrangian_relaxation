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

        # Compute and store modified edge weights
        self.modified_weights = self.compute_modified_weights()

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
        if self.branching_rule == "strong_branching":
            # Get all candidate edges that are not fixed or excluded
            candidate_edges = [(u, v) for (u, v, w, l) in self.edges if (u, v) not in self.fixed_edges and
                            (u, v) not in self.excluded_edges and
                            (u, v) not in self.branched_edges]

            if not candidate_edges:
                return None

            # Calculate the branching score for each candidate edge
            branching_scores = []
            for edge in candidate_edges:
                score = self.calculate_strong_branching_score(edge)
                branching_scores.append((edge, score))

            # Sort edges by their branching score (higher score is better)
            branching_scores.sort(key=lambda x: x[1], reverse=True)

            # Return the edge with the highest branching score
            return [branching_scores[0][0]] if branching_scores else None

        elif self.branching_rule == "most_fractional":
            # Compute the primal recovery solution
            shor_primal_solution = self.lagrangian_solver.compute_shor_primal_solution()
            if not shor_primal_solution:
                return None

            # Find the edge with the weight closest to 0.5
            most_fractional_edge = None
            min_distance = float('inf')

            for edge, weight in shor_primal_solution.items():
                distance = abs(weight - 0.5)
                if distance < min_distance:
                    min_distance = distance
                    most_fractional_edge = edge

            # Return the most fractional edge as the candidate
            return [most_fractional_edge] if most_fractional_edge else None

        elif self.branching_rule == "most_violated":
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
    
    def compute_modified_weights(self):
        """
        Compute and store the modified edge weights (w + lambda * l) for all edges.
        """
        modified_weights = {}
        for u, v, w, l in self.edges:
            modified_weights[(u, v)] = w 
            modified_weights[(v, u)] = w  # Store both directions
        return modified_weights
    
    def get_modified_weight(self, edge):
        """
        Get the modified weight of an edge using the current lambda value.
        """
        u, v = edge
        # Find the original weight and length of the edge
        w, l = next((w, l) for x, y, w, l in self.edges if (x, y) == (u, v) or (y, x) == (u, v))
        # Return the modified weight: w + lambda * l
        return w + self.lagrangian_solver.lmbda * l
    
    def calculate_strong_branching_score(self, edge):
        """
        Calculate the strong branching score for a given edge.
        """
        u, v = edge

        # Simulate fixing the edge
        fixed_lower_bound = self.simulate_fix_edge(u, v)
        if fixed_lower_bound == float('inf'):  # Infeasible case
            fix_score = float('inf')  # Assign a high penalty score
        else:
            fix_score = fixed_lower_bound - self.local_lower_bound

        print(f"Fix score for edge ({u}, {v}): {fix_score:.4f}")

        # Simulate excluding the edge
        excluded_lower_bound = self.simulate_exclude_edge(u, v)
        if excluded_lower_bound == float('inf'):  # Infeasible case
            exc_score = float('inf')  # Assign a high penalty score
        else:
            exc_score = excluded_lower_bound - self.local_lower_bound

        print(f"Exclude score for edge ({u}, {v}): {exc_score:.4f}")

        # Calculate the branching score as the maximum change in the lower bound
        if fix_score == float('inf') and exc_score == float('inf'):
            # Both fixing and excluding lead to infeasible solutions
            score = float('inf')  # Assign a high penalty score
        else:
            # Use the maximum of the absolute values of the feasible scores
            score = max(abs(fix_score) if fix_score != float('inf') else 0,
                        abs(exc_score) if exc_score != float('inf') else 0)

        return score

    def simulate_fix_edge(self, u, v):
        """
        Simulate fixing an edge in the MST and compute the new lower bound.
        """
        print(f"Simulating fixing edge ({u}, {v})")
        
        # Check if the edge is already in the MST
        if (u, v) in self.mst_edges or (v, u) in self.mst_edges:
            print("Edge already in MST, lower bound remains the same")
            return self.local_lower_bound

        # Create a copy of the MST and add the fixed edge to find the cycle
        mst_graph = nx.Graph(self.mst_edges)
        print("Edges in the MST graph:")
        for edge in mst_graph.edges():
            print(edge)

        total_weight = 0.0
        for edge in mst_graph.edges():
            total_weight += self.get_modified_weight(edge)
        print(f"Total weight of the MST based on modified weights: {total_weight:.5f}")
        print(f"budget{self.lagrangian_solver.lmbda * self.budget}")

        mst_graph.add_edge(u, v)

        # Find the cycle created by adding the edge (u, v)
        try:
            cycle = nx.find_cycle(mst_graph, source=u)
        except nx.NetworkXNoCycle:
            print("No cycle found, lower bound remains the same")
            return self.local_lower_bound

        # Exclude the fixed edge from the cycle
        cycle_without_fixed = [edge for edge in cycle if edge != (u, v) and edge != (v, u)]
        if not cycle_without_fixed:
            print("No other edges in the cycle, lower bound remains the same")
            return self.local_lower_bound

        # Find the heaviest edge in the cycle (excluding the fixed edge) that is not forced to be in the graph
        heaviest_edge = None
        max_weight = float('-inf')
        for edge in cycle_without_fixed:
            if edge not in self.fixed_edges:  # Ensure the edge is not forced to be in the graph
                edge_weight = self.get_modified_weight(edge)
                if edge_weight > max_weight:
                    max_weight = edge_weight
                    heaviest_edge = edge

        if not heaviest_edge:
            print("No valid edge to remove, pruning this branch (infeasible)")
            return float('inf')  # Prune this branch (infeasible)

        # Get the modified weight of the fixed edge
        fixed_edge_weight = self.get_modified_weight((u, v))
        print(f"Fixed edge ({u}, {v}): Modified Weight = {fixed_edge_weight:.5f}")


        # Get the modified weight of the heaviest edge in the cycle
        heaviest_edge_weight = self.get_modified_weight(heaviest_edge)
        print(f"Heaviest edge in cycle (excluding fixed edge): {heaviest_edge}, Modified Weight = {heaviest_edge_weight:.5f}")

        # Calculate the new lower bound
        new_lower_bound = self.local_lower_bound + fixed_edge_weight - heaviest_edge_weight
        print(f"New lower bound after fixing edge: {new_lower_bound:.5f}")
        print(f"Current lower bound: {self.local_lower_bound:.5f}")

        return new_lower_bound

    def simulate_exclude_edge(self, u, v):
        """
        Simulate excluding an edge from the MST and compute the new lower bound.
        """
        print(f"Simulating excluding edge ({u}, {v})")
        
        # Check if the edge is not in the MST
        if (u, v) not in self.mst_edges and (v, u) not in self.mst_edges:
            print("Edge not in MST, lower bound remains the same")
            return self.local_lower_bound

        # Create a copy of the MST and remove the excluded edge
        mst_graph = nx.Graph(self.mst_edges)
        mst_graph.remove_edge(u, v)

        # Find the two disconnected components
        components = list(nx.connected_components(mst_graph))
        if len(components) != 2:
            print("Error: Excluding the edge did not create exactly two components.")
            return float('inf')  # Prune this branch (infeasible)

        # Find the cheapest replacement edge that connects the two components
        cheapest_edge = None
        min_weight = float('inf')
        for x, y, w, l in self.edges:
            # Check if the edge connects the two components
            if (x in components[0] and y in components[1]) or (x in components[1] and y in components[0]):
                # Ensure the edge is not excluded by the parent nodes
                if (x, y) not in self.excluded_edges and (y, x) not in self.excluded_edges:
                    edge_weight = self.get_modified_weight((x, y))
                    if edge_weight < min_weight:
                        min_weight = edge_weight
                        cheapest_edge = (x, y)

        if not cheapest_edge:
            print("No replacement edge found, pruning this branch (infeasible)")
            return float('inf')  # Prune this branch (infeasible)

        # Get the modified weight of the excluded edge
        excluded_edge_weight = self.get_modified_weight((u, v))
        print(f"Excluded edge ({u}, {v}): Modified Weight = {excluded_edge_weight:.5f}")

        # Get the modified weight of the cheapest replacement edge
        replacement_edge_weight = self.get_modified_weight(cheapest_edge)
        print(f"Replacement edge {cheapest_edge}: Modified Weight = {replacement_edge_weight:.5f}")

        # Calculate the new lower bound
        new_lower_bound = self.local_lower_bound - excluded_edge_weight + replacement_edge_weight
        print(f"New lower bound after excluding edge: {new_lower_bound:.5f}")
        print(f"Current lower bound: {self.local_lower_bound:.5f}")

        return new_lower_bound
    
    def find_cycle_with_edge(self, u, v):
        """
        Find the cycle created by adding the edge (u, v) to the MST.
        """
        mst_graph = nx.Graph(self.mst_edges)
        mst_graph.add_edge(u, v)
        try:
            cycle = nx.find_cycle(mst_graph, source=u)
            return [(x, y) for x, y in cycle]
        except nx.NetworkXNoCycle:
            return None
    def find_cheapest_replacement_edge(self, u, v):
        """
        Find the cheapest edge to reconnect the MST after removing (u, v).
        The excluded edge (u, v) is not considered as a candidate for reconnection.
        """
        mst_graph = nx.Graph(self.mst_edges)
        mst_graph.remove_edge(u, v)

        # Find the two disconnected components
        components = list(nx.connected_components(mst_graph))
        if len(components) != 2:
            return None

        # Find the cheapest edge between the two components, excluding the removed edge (u, v)
        cheapest_edge = None
        min_weight = float('inf')

        for x, y, w, l in self.edges:
            # Ensure the edge is not the excluded edge
            if (x, y) == (u, v) or (y, x) == (u, v):
                continue

            # Check if the edge connects the two components
            if (x in components[0] and y in components[1]) or (x in components[1] and y in components[0]):
                if w < min_weight:
                    min_weight = w
                    cheapest_edge = (x, y)

        return cheapest_edge