import sys, argparse
import random
from mstkpbranchandbound import MSTNode
from branchandbound import RandomBranchingRule, BranchAndBound
from lagrangianrelaxation import LagrangianMST
from mstkpinstance import MSTKPInstance

def parse_arguments():
    # Create the parser
    parser = argparse.ArgumentParser(prog='MST Lagrangean B&B', usage='%(prog)s [options]')

    # Add the seed argument
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)"
    )

    # Add the number of nodes of the graph
    # The default value is 100
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=100,
        help="The number of nodes in the graph (default: 100)"
    )

    # Add the density of the graph
    # The default value is 0.3
    parser.add_argument(
        "--density",
        type=float,
        default=0.3,
        help="The density of the graph (default: 0.3)"
    )

    # Add the branching rule argument
    # The default is strong branching.
    parser.add_argument(
        "rule",
        choices=["random_mst", "random_all", "most_violated", "critical_edge", "most_fractional", "strong_branching"],  # Add "strong_branching"
        help="The branching rule to use (random_mst: pick from MST edges, random_all: pick from all variables, most_fractional: pick the most fractional edge, strong_branching: use strong branching)"
    )

    # Add a flag for inheriting lambda
    parser.add_argument(
        "--inherit-lambda",
        action="store_true",  # If the flag is provided, this will be True
        help="Inherit lambda from the parent node (default: False)"
    )

    # Add a flag for inheriting step size
    parser.add_argument(
        "--inherit-step-size",
        action="store_true",  # If the flag is provided, this will be True
        help="Inherit step size from the parent node (default: False)"
    )

    # Parse the arguments
    args = parser.parse_args()

    # Print the selected branching rule and inherit-lambda flag
    print(f"Using branching rule: {args.rule}")
    print(f"Inherit lambda: {args.inherit_lambda}")

    return args  # Return the args object


if __name__ == "__main__":

    # For debugging purposes, we set the branching rule to strong branching if it is not given as an argument
    # This is useful when we run the script from an IDE
    if len(sys.argv) == 1:
         sys.argv.append("strong_branching")

    # Call the function to parse arguments and store the result in a global variable
    args = parse_arguments()
    
    #initialize the random seed
    random.seed(args.seed)

    # Create an MSTKPInstance object
    mstkp_instance = MSTKPInstance(args.num_nodes, args.density)

    # Print all edges with their weight and length
    mstkp_instance.print_all_edges()

    root_node = MSTNode(mstkp_instance.edges, mstkp_instance.num_nodes, mstkp_instance.budget, initial_lambda=1.0, inherit_lambda=args.inherit_lambda, branching_rule=args.rule, step_size=1.0, inherit_step_size=args.inherit_step_size )
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