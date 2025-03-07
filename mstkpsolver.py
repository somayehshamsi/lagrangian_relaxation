import argparse
from mstkpbranchandbound import MSTNode
from branchandbound import RandomBranchingRule, BranchAndBound
from lagrangianrelaxation import LagrangianMST

def parse_arguments():
    # Create the parser
    parser = argparse.ArgumentParser(prog='MST Lagrangean B&B', usage='%(prog)s [options]')

    # Add the branching rule argument
    parser.add_argument(
        "rule",
        choices=["random_mst", "random_all"],  # Allow random_mst and random_all
        help="The branching rule to use (random_mst: pick from MST edges, random_all: pick from all variables)"
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

    # Call the function to parse arguments and store the result in a global variable
    args = parse_arguments()
    
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

    root_node = MSTNode(edges, num_nodes, budget, initial_lambda=1.0, inherit_lambda=args.inherit_lambda, branching_rule=args.rule, step_size=1.0, inherit_step_size=args.inherit_step_size )
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