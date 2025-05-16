

import heapq
import random
import networkx as nx
from lagrangianrelaxation import LagrangianMST
from branchandbound import Node, BranchAndBound, RandomBranchingRule

class MSTNode(Node):
    def __init__(self, edges, num_nodes, budget, fixed_edges=set(), excluded_edges=set(), branched_edges=set(), 
                 initial_lambda=0.1, inherit_lambda=False, branching_rule="random_mst",
                 step_size=0.01, inherit_step_size=False, use_cover_cuts=False, cut_frequency=5, 
                 node_cut_frequency=10, parent_cover_cuts=None, parent_cover_multipliers=None, 
                 use_bisection=False, max_iter=50, verbose=False):
        self.edges = edges
        self.num_nodes = num_nodes
        self.budget = budget
        self.fixed_edges = set(fixed_edges)
        self.excluded_edges = set(excluded_edges)
        self.branched_edges = set(branched_edges)

        self.inherit_lambda = inherit_lambda
        self.initial_lambda = initial_lambda if inherit_lambda else 0.1
        self.branching_rule = branching_rule
        self.step_size = step_size
        self.inherit_step_size = inherit_step_size

        self.use_cover_cuts = use_cover_cuts
        self.cut_frequency = cut_frequency
        self.node_cut_frequency = node_cut_frequency
        self.use_bisection = use_bisection
        self.verbose = verbose

        filtered_edges = [(u, v, w, l) for (u, v, w, l) in edges if (u, v) not in self.excluded_edges]

        self.lagrangian_solver = LagrangianMST(
            filtered_edges, num_nodes, budget, self.fixed_edges, self.excluded_edges,
            initial_lambda=self.initial_lambda, step_size=self.step_size,
            max_iter=max_iter, p=0.95,
            use_cover_cuts=self.use_cover_cuts,
            cut_frequency=self.cut_frequency,
            use_bisection=self.use_bisection,
            verbose=self.verbose
        )
        self.active_cuts = []
        self.cut_multipliers = {}
        if parent_cover_cuts and parent_cover_multipliers:
            for cut_idx, cut in enumerate(parent_cover_cuts):
                if not any((e in self.fixed_edges) for e in cut):
                    new_idx = len(self.active_cuts)
                    self.active_cuts.append(cut)
                    self.cut_multipliers[new_idx] = parent_cover_multipliers[cut_idx] * 0.7

        self.local_lower_bound, self.best_upper_bound, self.new_cuts = \
            self.lagrangian_solver.solve(
                inherited_cuts=self.active_cuts,
                inherited_multipliers=self.cut_multipliers
            )

        self.actual_cost, _ = self.lagrangian_solver.compute_real_weight_length()
        self.mst_edges = self.lagrangian_solver.last_mst_edges

        self.is_meaningful = False
        super().__init__(self.local_lower_bound)

    def __lt__(self, other):
        return self.local_lower_bound < other.local_lower_bound

    def is_child_likely_feasible(self):
        fixed_length = sum(
            self.lagrangian_solver.edge_attributes[(min(u, v), max(u, v))][1]
            for u, v in self.fixed_edges
        )
        num_fixed_edges = len(self.fixed_edges)
        num_nodes_covered = len(set(u for u, v in self.fixed_edges) | set(v for u, v in self.fixed_edges))
        components = max(1, num_nodes_covered - num_fixed_edges + 1)
        edges_needed = self.num_nodes - num_nodes_covered + components - 1

        if edges_needed < 0:
            return False

        min_edge_length = float('inf')
        for u, v, _, l in self.edges:
            if (u, v) not in self.excluded_edges and (v, u) not in self.excluded_edges:
                min_edge_length = min(min_edge_length, l)

        estimated_length = fixed_length + edges_needed * min_edge_length
        return estimated_length <= self.budget

    def create_children(self, branched_edge):
        u, v = branched_edge[0], branched_edge[1]
        new_branched_edges = self.branched_edges | {(u, v)}

        all_cuts = self.active_cuts + self.new_cuts
        capped_multipliers = {}
        for new_idx, cut in enumerate(all_cuts):
            parent_idx = None
            if cut in self.active_cuts:
                parent_idx = self.active_cuts.index(cut)
            elif cut in self.new_cuts:
                parent_idx = len(self.active_cuts) + self.new_cuts.index(cut)
                
            if parent_idx is not None and parent_idx in self.cut_multipliers:
                capped_multipliers[new_idx] = self.cut_multipliers[parent_idx] * 0.7
            else:
                capped_multipliers[new_idx] = 1.0

        current_lambda = self.lagrangian_solver.best_lambda if self.inherit_lambda else 0.1
        current_step_size = self.lagrangian_solver.step_size if self.inherit_step_size else 0.01

        fixed_child = MSTNode(self.edges, self.num_nodes, self.budget,
                            self.fixed_edges | {(u, v)}, self.excluded_edges, new_branched_edges,
                            initial_lambda=current_lambda, inherit_lambda=self.inherit_lambda,
                            branching_rule=self.branching_rule, step_size=current_step_size,
                            inherit_step_size=self.inherit_step_size,
                            use_cover_cuts=self.use_cover_cuts,
                            cut_frequency=self.cut_frequency,
                            node_cut_frequency=self.node_cut_frequency,
                            parent_cover_cuts=all_cuts,
                            parent_cover_multipliers=capped_multipliers,
                            use_bisection=self.use_bisection,
                            max_iter=50,
                            verbose=self.verbose)

        excluded_child = MSTNode(self.edges, self.num_nodes, self.budget,
                                self.fixed_edges, self.excluded_edges | {(u, v)}, new_branched_edges,
                                initial_lambda=current_lambda, inherit_lambda=self.inherit_lambda,
                                branching_rule=self.branching_rule, step_size=current_step_size,
                                inherit_step_size=self.inherit_step_size,
                                use_cover_cuts=self.use_cover_cuts,
                                cut_frequency=self.cut_frequency,
                                node_cut_frequency=self.node_cut_frequency,
                                parent_cover_cuts=all_cuts,
                                parent_cover_multipliers=capped_multipliers,
                                use_bisection=self.use_bisection,
                                max_iter=50,
                                verbose=self.verbose)
        return [fixed_child, excluded_child]

    def is_feasible(self):
        real_weight, real_length = self.lagrangian_solver.compute_real_weight_length()
        if real_length > self.budget:
            return False, "MST length exceeds budget"

        mst_nodes = set()
        for u, v in self.mst_edges:
            mst_nodes.add(u)
            mst_nodes.add(v)

        if len(mst_nodes) < self.num_nodes:
            return False, "MST does not include all nodes"

        mst_graph = nx.Graph(self.mst_edges)
        if not nx.is_connected(mst_graph):
            return False, "MST is not connected"

        return True, "MST is feasible"

    def compute_upper_bound(self):
        real_weight, _ = self.lagrangian_solver.compute_real_weight_length()
        return real_weight

    def get_branching_candidates(self):
        if self.branching_rule == "strong_branching":
            mst_edges = self.lagrangian_solver.best_mst_edges
            candidate_edges = [
                e for e in mst_edges
                if (e[0], e[1]) not in self.fixed_edges and
                   (e[0], e[1]) not in self.excluded_edges and
                   (e[0], e[1]) not in self.branched_edges
            ]
            if not candidate_edges:
                return None

            if self.verbose:
                print(f"Node {id(self)}: Strong branching evaluating {len(candidate_edges)} MST edges: {candidate_edges}")
            
            best_edge = None
            best_score = -float('inf')
            for edge in candidate_edges:
                score = self.calculate_strong_branching_score(edge)
                if self.verbose:
                    print(f"Edge {edge}: Score {score}")
                if score > best_score:
                    best_score = score
                    best_edge = edge

            return [best_edge] if best_edge else None

        elif self.branching_rule == "strong_branching_sim":
            mst_edges = self.lagrangian_solver.best_mst_edges
            candidate_edges = [
                e for e in mst_edges
                if (e[0], e[1]) not in self.fixed_edges and
                   (e[0], e[1]) not in self.excluded_edges and
                   (e[0], e[1]) not in self.branched_edges
            ]
            if not candidate_edges:
                return None

            if self.verbose:
                print(f"Node {id(self)}: Strong branching sim evaluating {len(candidate_edges)} MST edges: {candidate_edges}")
            
            best_edge = None
            best_score = -float('inf')
            for edge in candidate_edges:
                u, v = edge
                fixed_lower_bound = self.simulate_fix_edge(u, v)
                fix_score = (fixed_lower_bound - self.local_lower_bound) if fixed_lower_bound != float('inf') else float('inf')
                excluded_lower_bound = self.simulate_exclude_edge(u, v)
                exc_score = (excluded_lower_bound - self.local_lower_bound) if excluded_lower_bound != float('inf') else float('inf')
                score = min(fix_score, exc_score) + 0.1 * max(fix_score, exc_score) if fix_score != float('inf') and exc_score != float('inf') else float('inf')
                if self.verbose:
                    print(f"Edge {edge}: Score {score}")
                if score > best_score:
                    best_score = score
                    best_edge = edge

            return [best_edge] if best_edge else None

        elif self.branching_rule in ["most_fractional", "sb_fractional"]:
            shor_primal_solution = self.lagrangian_solver.compute_dantzig_wolfe_solution(self)
            if not shor_primal_solution:
                if self.verbose:
                    print("No Dantzig-Wolfe solution; falling back to MST edges")
                # Fallback 1: Use valid edges from best_mst_edges
                valid_edges = {(min(u, v), max(u, v)) for u, v, _, _ in self.edges}
                candidates = [
                    e for e in self.lagrangian_solver.best_mst_edges
                    if (e not in self.fixed_edges) and 
                       (e not in self.excluded_edges) and
                       ((e[1], e[0]) not in self.fixed_edges) and
                       ((e[1], e[0]) not in self.excluded_edges) and
                       e not in self.branched_edges and
                       (min(e[0], e[1]), max(e[0], e[1])) in valid_edges
                ]
                if not candidates:
                    if self.verbose:
                        print("No best MST candidates; using all valid edges")
                    # Fallback 2: Use all valid graph edges
                    candidates = [
                        (u, v) for u, v, w, l in self.edges
                        if (u, v) not in self.fixed_edges and 
                           (u, v) not in self.excluded_edges and
                           (v, u) not in self.fixed_edges and
                           (v, u) not in self.excluded_edges and
                           (u, v) not in self.branched_edges
                    ]
                if not candidates:
                    if self.verbose:
                        print("No branching candidates available")
                    return None
                # For most_fractional, select edge closest to 0.5 in best MST (arbitrary for fallback)
                # For sb_fractional, use strong branching score
                if self.branching_rule == "most_fractional":
                    return [candidates[0]] if candidates else None
                else:  # sb_fractional
                    best_edge = None
                    best_score = -float('inf')
                    for edge in candidates:
                        score = self.calculate_strong_branching_score(edge)
                        if score > best_score:
                            best_score = score
                            best_edge = edge
                    return [best_edge] if best_edge else None

            # Normalize edges and validate against graph edges
            valid_edges = {(min(u, v), max(u, v)) for u, v, _, _ in self.edges}
            normalized_edge_weights = {}
            for e in shor_primal_solution:
                normalized_e = (min(e[0], e[1]), max(e[0], e[1]))
                if normalized_e in valid_edges:
                    normalized_edge_weights[normalized_e] = shor_primal_solution[e]
                else:
                    if self.verbose:
                        print(f"Warning: Edge {e} from Dantzig-Wolfe not in graph edges; skipping")

            # Candidates: edges that are fractional (not exactly 0 or 1) and not fixed/excluded
            candidates = [
                e for e in normalized_edge_weights
                if (e not in self.fixed_edges) and 
                   (e not in self.excluded_edges) and
                   ((e[1], e[0]) not in self.fixed_edges) and
                   ((e[1], e[0]) not in self.excluded_edges) and
                   abs(normalized_edge_weights[e]) > 1e-6 and 
                   abs(normalized_edge_weights[e] - 1.0) > 1e-6
            ]

            if not candidates:
                if self.verbose:
                    print("No fractional branching candidates; using valid best MST edges")
                # Fallback 1: Use valid edges from best_mst_edges
                candidates = [
                    e for e in self.lagrangian_solver.best_mst_edges
                    if (e not in self.fixed_edges) and 
                       (e not in self.excluded_edges) and
                       ((e[1], e[0]) not in self.fixed_edges) and
                       ((e[1], e[0]) not in self.excluded_edges) and
                       e not in self.branched_edges and
                       (min(e[0], e[1]), max(e[0], e[1])) in valid_edges
                ]
                if not candidates:
                    if self.verbose:
                        print("No best MST candidates; using all valid edges")
                    # Fallback 2: Use all valid graph edges
                    candidates = [
                        (u, v) for u, v, w, l in self.edges
                        if (u, v) not in self.fixed_edges and 
                           (u, v) not in self.excluded_edges and
                           (v, u) not in self.fixed_edges and
                           (v, u) not in self.excluded_edges and
                           (u, v) not in self.branched_edges
                    ]
                if not candidates:
                    if self.verbose:
                        print("No branching candidates available")
                    return None

            # Compute scores for candidates
            branching_scores = []
            edge_attributes = {(min(u, v), max(u, v)): (w, l) for u, v, w, l in self.edges}
            for e in candidates:
                try:
                    w = normalized_edge_weights.get(e, 0)
                    if w == 0 and e not in self.lagrangian_solver.best_mst_edges:
                        if self.verbose:
                            print(f"Warning: Edge {e} has zero weight and not in best MST; skipping")
                        continue
                    if self.branching_rule == "most_fractional":
                        # Primary: distance from 0.5 (closer to 0.5 is better)
                        distance_score = -abs(w - 0.5)
                        # Secondary: prefer weights closer to 1
                        weight_preference = w
                        # Tertiary: edge weight (impact on objective)
                        edge_weight = edge_attributes.get(e, (0, 0))[0]
                        weight_score = -edge_weight
                        # Quaternary: edge length (impact on budget)
                        edge_length = edge_attributes.get(e, (0, 0))[1]
                        length_score = -edge_length
                        composite_score = (distance_score, weight_preference, weight_score, length_score)
                        branching_scores.append((e, composite_score))
                    else:  # sb_fractional
                        score = self.calculate_strong_branching_score(e)
                        branching_scores.append((e, score))
                except KeyError as ke:
                    if self.verbose:
                        print(f"Error: Edge {e} not found during scoring; skipping")
                    continue

            if not branching_scores:
                if self.verbose:
                    print("No valid branching scores computed; using fallback")
                # Fallback for scoring failure
                candidates = [
                    e for e in self.lagrangian_solver.best_mst_edges
                    if (e not in self.fixed_edges) and 
                       (e not in self.excluded_edges) and
                       ((e[1], e[0]) not in self.fixed_edges) and
                       ((e[1], e[0]) not in self.excluded_edges) and
                       e not in self.branched_edges and
                       (min(e[0], e[1]), max(e[0], e[1])) in valid_edges
                ]
                if not candidates:
                    candidates = [
                        (u, v) for u, v, w, l in self.edges
                        if (u, v) not in self.fixed_edges and 
                           (u, v) not in self.excluded_edges and
                           (v, u) not in self.fixed_edges and
                           (v, u) not in self.excluded_edges and
                           (u, v) not in self.branched_edges
                    ]
                if not candidates:
                    return None
                if self.branching_rule == "most_fractional":
                    return [candidates[0]] if candidates else None
                else:  # sb_fractional
                    best_edge = None
                    best_score = -float('inf')
                    for edge in candidates:
                        score = self.calculate_strong_branching_score(edge)
                        if score > best_score:
                            best_score = score
                            best_edge = edge
                    return [best_edge] if best_edge else None

            if self.verbose:
                print(f"Node {id(self)}: {self.branching_rule} evaluating {len(branching_scores)} edges")
                for edge, score in branching_scores:
                    if self.branching_rule == "most_fractional":
                        print(f"Edge {edge}: Composite score=(distance={score[0]:.6f}, weight_pref={score[1]:.6f}, "
                              f"weight={score[2]:.6f}, length={score[3]:.6f})")
                    else:
                        print(f"Edge {edge}: Strong branching score={score:.6f}")

            # Sort by score (higher is better)
            if self.branching_rule == "most_fractional":
                branching_scores.sort(key=lambda x: x[1], reverse=True)
            else:  # sb_fractional
                branching_scores.sort(key=lambda x: x[1], reverse=True)
            return [branching_scores[0][0]] if branching_scores else None

        elif self.branching_rule == "most_violated":
            candidate_edges = sorted(
                [(u, v, w, l) for u, v, w, l in self.edges if (u, v) not in self.fixed_edges and (u, v) not in self.excluded_edges],
                key=lambda x: x[2] / x[3],
                reverse=True,
            )
        elif self.branching_rule == "random_mst":
            assert self.mst_edges
            candidate_edges = [e for e in self.mst_edges if (e[0], e[1]) not in self.fixed_edges and
                              (e[0], e[1]) not in self.excluded_edges and
                              (e[0], e[1]) not in self.branched_edges]
        elif self.branching_rule == "random_all":
            candidate_edges = [(u, v) for (u, v, w, l) in self.edges if (u, v) not in self.fixed_edges and
                              (u, v) not in self.excluded_edges and
                              (u, v) not in self.branched_edges]
        else:
            raise ValueError(f"Unknown branching rule: {self.branching_rule}")
        return candidate_edges if candidate_edges else None

    def get_modified_weight(self, edge):
        u, v = edge
        w, l = next((w, l) for x, y, w, l in self.edges 
                    if (x,y) == (u,v) or (y,x) == (u,v))
        modified = w + self.lagrangian_solver.best_lambda * l
        
        for cut_idx, cut in enumerate(self.active_cuts):
            if (u,v) in cut or (v,u) in cut:
                modified += self.cut_multipliers.get(cut_idx, 0)
        
        return modified
    
    def calculate_strong_branching_score(self, edge):
        if self.branching_rule == "strong_branching":
            u, v = edge
            all_cuts = self.active_cuts + self.new_cuts
            capped_multipliers = {}
            for new_idx, cut in enumerate(all_cuts):
                parent_idx = None
                if cut in self.active_cuts:
                    parent_idx = self.active_cuts.index(cut)
                elif cut in self.new_cuts:
                    parent_idx = len(self.active_cuts) + self.new_cuts.index(cut)
                if parent_idx is not None and parent_idx in self.cut_multipliers:
                    capped_multipliers[new_idx] = self.cut_multipliers[parent_idx] * 0.7
                else:
                    capped_multipliers[new_idx] = 1.0

            current_lambda = self.lagrangian_solver.best_lambda if self.inherit_lambda else 0.1
            current_step_size = self.lagrangian_solver.step_size if self.inherit_step_size else 0.01

            fixed_child = MSTNode(
                self.edges,
                self.num_nodes,
                self.budget,
                fixed_edges=self.fixed_edges | {(u, v)},
                excluded_edges=self.excluded_edges,
                branched_edges=self.branched_edges | {(u, v)},
                initial_lambda=current_lambda,
                inherit_lambda=self.inherit_lambda,
                branching_rule=self.branching_rule,
                step_size=current_step_size,
                inherit_step_size=self.inherit_step_size,
                use_cover_cuts=self.use_cover_cuts,
                cut_frequency=self.cut_frequency,
                node_cut_frequency=self.node_cut_frequency,
                parent_cover_cuts=all_cuts,
                parent_cover_multipliers=capped_multipliers,
                use_bisection=self.use_bisection,
                max_iter=30,
                verbose=self.verbose
            )
            fixed_lower_bound = fixed_child.local_lower_bound
            fixed_upper_bound = fixed_child.best_upper_bound

            excluded_child = MSTNode(
                self.edges,
                self.num_nodes,
                self.budget,
                fixed_edges=self.fixed_edges,
                excluded_edges=self.excluded_edges | {(u, v)},
                branched_edges=self.branched_edges | {(u, v)},
                initial_lambda=current_lambda,
                inherit_lambda=self.inherit_lambda,
                branching_rule=self.branching_rule,
                step_size=current_step_size,
                inherit_step_size=self.inherit_step_size,
                use_cover_cuts=self.use_cover_cuts,
                cut_frequency=self.cut_frequency,
                node_cut_frequency=self.node_cut_frequency,
                parent_cover_cuts=all_cuts,
                parent_cover_multipliers=capped_multipliers,
                use_bisection=self.use_bisection,
                max_iter=30,
                verbose=self.verbose
            )
            excluded_lower_bound = excluded_child.local_lower_bound
            excluded_upper_bound = excluded_child.best_upper_bound

            fix_lower_score = (fixed_lower_bound - self.local_lower_bound) if fixed_lower_bound != float('inf') else float('inf')
            exc_lower_score = (excluded_lower_bound - self.local_lower_bound) if excluded_lower_bound != float('inf') else float('inf')
            
            fix_score = fix_lower_score
            exc_score = exc_lower_score

            if fix_score == float('inf') and exc_score == float('inf'):
                score = float('inf')
            else:
                score = 0.9 * min(fix_score, exc_score) + 0.1 * max(fix_score, exc_score)
            return score

        else:
            u, v = edge
            fixed_lower_bound = self.simulate_fix_edge(u, v)
            if fixed_lower_bound == float('inf'):
                fix_score = float('inf')
            else:
                fix_score = fixed_lower_bound - self.local_lower_bound

            excluded_lower_bound = self.simulate_exclude_edge(u, v)
            if excluded_lower_bound == float('inf'):
                exc_score = float('inf')
            else:
                exc_score = excluded_lower_bound - self.local_lower_bound

            if fix_score == float('inf') and exc_score == float('inf'):
                score = float('inf')
            else:
                score = min(fix_score, exc_score) + 0.1 * max(fix_score, exc_score)
            return score

    def simulate_fix_edge(self, u, v):
        mst_edges = self.lagrangian_solver.best_mst_edges
        if (u, v) in mst_edges or (v, u) in mst_edges:
            return self.local_lower_bound

        mst_graph = nx.Graph(mst_edges)
        mst_graph.add_edge(u, v)

        try:
            cycle = nx.find_cycle(mst_graph, source=u)
        except nx.NetworkXNoCycle:
            return self.local_lower_bound

        cycle_without_fixed = [edge for edge in cycle if edge != (u, v) and edge != (v, u)]
        heaviest_edge = None
        max_weight = float('-inf')
        for edge in cycle_without_fixed:
            if edge not in self.fixed_edges and (edge[1], edge[0]) not in self.fixed_edges:
                edge_weight = self.get_modified_weight(edge)
                if edge_weight > max_weight:
                    max_weight = edge_weight
                    heaviest_edge = edge

        if not heaviest_edge:
            return float('inf')

        fixed_edge_weight = self.get_modified_weight((u, v))
        heaviest_edge_weight = self.get_modified_weight(heaviest_edge)
        new_lower_bound = self.local_lower_bound + fixed_edge_weight - heaviest_edge_weight
        return new_lower_bound

    def simulate_exclude_edge(self, u, v):
        mst_edges = self.lagrangian_solver.best_mst_edges
        if (u, v) not in mst_edges and (v, u) not in mst_edges:
            return self.local_lower_bound

        mst_graph = nx.Graph(mst_edges)
        mst_graph.remove_edge(u, v)

        components = list(nx.connected_components(mst_graph))
        if len(components) != 2:
            return float('inf')

        cheapest_edge = None
        min_weight = float('inf')
        for x, y, w, l in self.edges:
            if (x, y) == (u, v) or (y, x) == (u, v):
                continue
            if (x in components[0] and y in components[1]) or (x in components[1] and y in components[0]):
                if (x, y) not in self.excluded_edges and (y, x) not in self.excluded_edges:
                    edge_weight = self.get_modified_weight((x, y))
                    if edge_weight < min_weight:
                        min_weight = edge_weight
                        cheapest_edge = (x, y)

        if not cheapest_edge:
            return float('inf')

        excluded_edge_weight = self.get_modified_weight((u, v))
        replacement_edge_weight = self.get_modified_weight(cheapest_edge)
        new_lower_bound = self.local_lower_bound - excluded_edge_weight + replacement_edge_weight
        return new_lower_bound
    
    def print_cut_info(self):
        if self.verbose:
            print(f"\nNode Cut Status (Fixed: {self.fixed_edges}, Excluded: {self.excluded_edges})")
            print("Active Cuts:")
            for i, cut in enumerate(self.active_cuts):
                mult = self.cut_multipliers.get(i, 0)
                print(f"Cut {i}: {cut} (Multiplier: {mult:.3f})")
            
            print("\nInherited Cuts Breakdown:")
            inherited_from_parent = 0
            new_generated = 0
            for cut in self.lagrangian_solver.best_cuts:
                if cut in self.active_cuts:
                    inherited_from_parent += 1
                else:
                    new_generated += 1
            print(f"Total cuts: {len(self.lagrangian_solver.best_cuts)}")
            print(f" - Inherited: {inherited_from_parent}")
            print(f" - New: {new_generated}")