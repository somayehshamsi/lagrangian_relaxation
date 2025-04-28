import sys, argparse
import random
from mstkpbranchandbound import MSTNode
from branchandbound import RandomBranchingRule, BranchAndBound
from lagrangianrelaxation import LagrangianMST
from mstkpinstance import MSTKPInstance

def parse_arguments():
    parser = argparse.ArgumentParser(prog='MST Lagrangean B&B', usage='%(prog)s [options]')
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)"
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=100,
        help="The number of nodes in the graph (default: 100)"
    )
    parser.add_argument(
        "--density",
        type=float,
        default=0.3,
        help="The density of the graph (default: 0.3)"
    )
    parser.add_argument(
        "rule",
        choices=["random_mst", "random_all", "most_violated", "critical_edge", "most_fractional", "strong_branching"],
        help="The branching rule to use (random_mst: pick from MST edges, random_all: pick from all variables, most_fractional: pick the most fractional edge, strong_branching: use strong branching)"
    )
    parser.add_argument(
        "--inherit-lambda",
        action="store_true",
        help="Inherit lambda from the parent node (default: False)"
    )
    parser.add_argument(
        "--inherit-step-size",
        action="store_true",
        help="Inherit step size from the parent node (default: False)"
    )
    parser.add_argument(
        "--cover-cuts",
        action="store_true",
        help="Enable cover cuts generation (default: False)"
    )
    parser.add_argument(
        "--cut-frequency",
        type=int,
        default=5,
        help="Frequency of cut generation in Lagrangian iterations (default: 5)"
    )
    parser.add_argument(
        "--node-cut-frequency",
        type=int,
        default=10,
        help="Frequency of cut generation in B&B nodes (default: 10)"
    )
    parser.add_argument(
        "--use-bisection",
        action="store_true",
        help="Use bisection algorithm for updating the knapsack multiplier (default: False)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debug output (default: False)"
    )

    args = parser.parse_args()
    print(f"Using branching rule: {args.rule}")
    print(f"Inherit lambda: {args.inherit_lambda}")
    print(f"Verbose: {args.verbose}")
    return args

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.append("strong_branching")

    args = parse_arguments()
    random.seed(args.seed)

    mstkp_instance = MSTKPInstance(args.num_nodes, args.density)
    root_node = MSTNode(mstkp_instance.edges, mstkp_instance.num_nodes, mstkp_instance.budget, initial_lambda=0.1, inherit_lambda=args.inherit_lambda, branching_rule=args.rule, step_size=0.005, inherit_step_size=args.inherit_step_size,
                        use_cover_cuts=args.cover_cuts, cut_frequency=args.cut_frequency, node_cut_frequency=10, parent_cover_cuts=None, parent_cover_multipliers=None, use_bisection=args.use_bisection)
    
    branching_rule = RandomBranchingRule()
    bnb_solver = BranchAndBound(branching_rule, verbose=args.verbose)
    best_solution, best_upper_bound = bnb_solver.solve(root_node)
    
    print(f"Optimal MST Cost within Budget: {best_upper_bound}")
    if best_solution:
        print("Edges in the Optimal MST:")
        for edge in best_solution.mst_edges:
            print(edge)
    else:
        print("No feasible solution found.")

    print(f"Lagrangian MST time: {LagrangianMST.total_compute_time:.2f}s")



