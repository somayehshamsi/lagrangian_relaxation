
import os
import random
import argparse
import pandas as pd
import numpy as np
from time import time
from mstkpinstance import MSTKPInstance
from mstkpbranchandbound import MSTNode
from branchandbound import RandomBranchingRule, BranchAndBound
from lagrangianrelaxation import LagrangianMST
import pickle
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(prog='MSTKP Benchmark Comparison', usage='%(prog)s [options]')
    parser.add_argument(
        "--num-instances",
        type=int,
        default=5,
        help="Number of instances to generate (default: 5)"
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=50,
        help="Number of nodes in each graph (default: 50)"
    )
    parser.add_argument(
        "--density",
        type=float,
        default=0.3,
        help="Edge density of the graph (default: 0.3)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/Users/ssha0224/Desktop",
        help="Directory to save results and instances (default: /Users/ssha0224/Desktop)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output (default: False)"
    )
    return parser.parse_args()

def generate_instances(num_instances, num_nodes, density, seed):
    random.seed(seed)
    instances = []
    for i in range(num_instances):
        instance_seed = random.randint(0, 1000000)
        random.seed(instance_seed)
        instance = MSTKPInstance(num_nodes, density)
        instances.append((instance, instance_seed))
        logger.info(f"Generated instance {i+1}/{num_instances} with seed {instance_seed}")
    return instances

def save_instances(instances, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    instances_path = os.path.join(output_dir, "instances.pkl")
    with open(instances_path, 'wb') as f:
        pickle.dump(instances, f)
    logger.info(f"Saved instances to {instances_path}")

def load_instances(output_dir):
    instances_path = os.path.join(output_dir, "instances.pkl")
    if not os.path.exists(instances_path):
        raise FileNotFoundError(f"No instances found at {instances_path}")
    with open(instances_path, 'rb') as f:
        instances = pickle.load(f)
    logger.info(f"Loaded {len(instances)} instances from {instances_path}")
    return instances

def run_experiment(instance, seed, config, args):
    random.seed(seed)
    mstkp_instance = instance
    branching_rule = config["branching_rule"]
    use_bisection = config["use_bisection"]
    use_2opt = config["use_2opt"]
    use_shooting = config["use_shooting"]
    cover_cuts = config["cover_cuts"]
    inherit_lambda = config["inherit_lambda"]

    root_node = MSTNode(
        mstkp_instance.edges,
        mstkp_instance.num_nodes,
        mstkp_instance.budget,
        initial_lambda=0.1,
        inherit_lambda=inherit_lambda,
        branching_rule=branching_rule,
        step_size=0.01,
        inherit_step_size=False,
        use_cover_cuts=cover_cuts,
        cut_frequency=5,
        node_cut_frequency=10,
        parent_cover_cuts=None,
        parent_cover_multipliers=None,
        use_bisection=use_bisection,
        verbose=args.verbose
    )

    branching_rule_obj = RandomBranchingRule()
    bnb_solver = BranchAndBound(
        branching_rule_obj,
        verbose=args.verbose,
        config=config,
        instance_seed=seed
    )

    two_opt_time = 0.0
    initial_solution = None
    initial_upper_bound = float("inf")

    if use_2opt:
        from mstkpsolver import two_opt_local_search
        start_2opt = time()
        initial_solution, initial_upper_bound = two_opt_local_search(
            mstkp_instance.edges,
            mstkp_instance.num_nodes,
            mstkp_instance.budget,
            root_node.mst_edges,
            verbose=args.verbose
        )
        two_opt_time = time() - start_2opt

    start_bnb = time()
    if use_shooting:
        best_solution, best_upper_bound = bnb_solver.solve_with_shooting(
            root_node,
            initial_lower_bound=root_node.local_lower_bound,
            initial_upper_bound=initial_upper_bound,
            initial_solution=initial_solution if initial_solution else None
        )
    else:
        best_solution, best_upper_bound = bnb_solver.solve(root_node)
    bnb_time = time() - start_bnb

    lagrangian_time = LagrangianMST.total_compute_time
    total_time = bnb_time + lagrangian_time + two_opt_time

    result = {
        "instance_seed": seed,
        "num_nodes": mstkp_instance.num_nodes,
        "density": mstkp_instance.density,
        "budget": mstkp_instance.budget,
        "branching_rule": branching_rule,
        "use_2opt": use_2opt,
        "use_shooting": use_shooting,
        "use_bisection": use_bisection,
        "cover_cuts": cover_cuts,
        "inherit_lambda": inherit_lambda,
        "total_time": total_time,
        "bnb_time": bnb_time,
        "lagrangian_time": lagrangian_time,
        "two_opt_time": two_opt_time,
        "total_nodes_solved": bnb_solver.total_nodes_solved
    }
    return result

def analyze_results(results):
    df = pd.DataFrame(results)
    summary = df.groupby(["branching_rule", "use_bisection", "use_2opt", "use_shooting", "cover_cuts", "inherit_lambda"]).agg({
        "total_time": ["mean", "std"],
        "bnb_time": ["mean", "std"],
        "lagrangian_time": ["mean", "std"],
        "total_nodes_solved": ["mean", "std"]
    }).round(2)
    return summary

def main():
    args = parse_arguments()
    random.seed(args.seed)

    output_dir = os.path.join(args.output_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(output_dir, exist_ok=True)

    # Delete existing instances.pkl to ensure new instances are generated
    instances_path = os.path.join(args.output_dir, "instances.pkl")
    if os.path.exists(instances_path):
        os.remove(instances_path)
        logger.info(f"Deleted existing {instances_path} to generate new instances")

    # Generate instances
    instances = generate_instances(args.num_instances, args.num_nodes, args.density, args.seed)
    save_instances(instances, args.output_dir)

    # Define configurations with varied cover_cuts and inherit_lambda
    configs = [
        {
            "branching_rule": "most_fractional",
            "use_bisection": False,
            "use_2opt": False,
            "use_shooting": False,
            "cover_cuts": True,
            "inherit_lambda": True
        },
        {
            "branching_rule": "strong_branching",
            "use_bisection": False,
            "use_2opt": False,
            "use_shooting": False,
            "cover_cuts": True,
            "inherit_lambda": True
        },
        {
            "branching_rule": "random_mst",
            "use_bisection": False,
            "use_2opt": True,
            "use_shooting": True,
            "cover_cuts": True,
            "inherit_lambda": False
        },
        {
            "branching_rule": "sb_fractional",
            "use_bisection": False,
            "use_2opt": False,
            "use_shooting": False,
            "cover_cuts": True,
            "inherit_lambda": True
        }
    ]

    results = []
    for config_idx, config in enumerate(configs):
        branching_rule = config["branching_rule"]
        logger.info(f"Running configuration {config_idx+1}/{len(configs)}: "
                    f"Branching={branching_rule}, Bisection={config['use_bisection']}, "
                    f"2-opt={config['use_2opt']}, Shooting={config['use_shooting']}, "
                    f"Cover Cuts={config['cover_cuts']}, Inherit Lambda={config['inherit_lambda']}")
        for instance_idx, (instance, instance_seed) in enumerate(instances):
            logger.info(f"Processing instance {instance_idx+1}/{len(instances)} with seed {instance_seed}")
            LagrangianMST.total_compute_time = 0.0  # Reset compute time
            try:
                result = run_experiment(instance, instance_seed, config, args)
                results.append(result)
                logger.info(f"Completed instance {instance_idx+1}: Total Time={result['total_time']:.2f}s, "
                            f"BNB Time={result['bnb_time']:.2f}s, Lagrangian Time={result['lagrangian_time']:.2f}s, "
                            f"Nodes Solved={result['total_nodes_solved']}")
            except Exception as e:
                logger.error(f"Error processing instance {instance_idx+1}: {str(e)}")
                continue

        # Save intermediate results
        df = pd.DataFrame(results)
        intermediate_path = os.path.join(output_dir, f"results_intermediate_{branching_rule}.csv")
        df.to_csv(intermediate_path, index=False)
        logger.info(f"Saved intermediate results to {intermediate_path}")

    # Save final results
    final_path = os.path.join(output_dir, "results.csv")
    df = pd.DataFrame(results)
    df.to_csv(final_path, index=False)
    logger.info(f"Saved final results to {final_path}")

    # Analyze results
    summary = analyze_results(results)
    summary_path = os.path.join(output_dir, "summary.csv")
    summary.to_csv(summary_path)
    logger.info(f"Saved summary statistics to {summary_path}")
    print("\nSummary Statistics:")
    print(summary)

if __name__ == "__main__":
    main()