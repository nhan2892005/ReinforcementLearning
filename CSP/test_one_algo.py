import gym_cutting_stock
import gymnasium as gym
from policy import GreedyPolicy, RandomPolicy
from student_submissions.s2210xxx.policy2210xxx \
import (
    Policy2210xxx as GreedySorted,
    BestFit,
    DynamicProgramming,
    GenerateColumn
)
import matplotlib.pyplot as plt
import sys

plt.ion()

# List of policy classes
policy_list = [
    GreedyPolicy,
    RandomPolicy,
    GreedySorted,
    BestFit,
    DynamicProgramming,
    GenerateColumn
]

def print_usage():
    print("Usage: python test_one_algo.py [index]")
    print("\nAvailable policies:")
    for idx, policy in enumerate(policy_list):
        print(f"{idx}: {policy.__name__}")

def main(index_of_policy=0):
    # Get the selected policy class
    try:
        SelectedPolicy = policy_list[index_of_policy]
    except IndexError:
        print(f"Invalid index {index_of_policy}. Please choose an index between 0 and {len(policy_list)-1}.")
        print_usage()
        return

    env = gym.make(
        "gym_cutting_stock/CuttingStock-v0",
        render_mode="human",
        min_w=20,
        min_h=20,
        max_w=50,
        max_h=50,
        num_stocks=100,
        max_product_type=25,
        max_product_per_type=20,
    )
    observation, info = env.reset(seed=42)
    test_policy = SelectedPolicy()
    for _ in range(1000):
        prod = observation["products"]
        print(f'Number of products: {sum([p["quantity"] for p in prod])}')
        action = test_policy.get_action(observation, info)
        print(action)
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            observation, info = env.reset()
            break

if __name__ == "__main__":
    # Check if an index was passed as a command-line argument
    if len(sys.argv) > 1:
        if sys.argv[1] in ("-h", "--help"):
            print_usage()
            sys.exit(0)
        try:
            index = int(sys.argv[1])
        except ValueError:
            print("Please provide a valid integer index.")
            print_usage()
            sys.exit(1)
        main(index_of_policy=index)
    else:
        # Default to index 0 if no argument is provided
        main()