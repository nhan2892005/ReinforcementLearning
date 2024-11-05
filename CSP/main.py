import gym_cutting_stock
import gymnasium as gym
import numpy as np
from policy import GreedyPolicy, RandomPolicy
from student_submissions.s2210xxx.policy2210xxx import Policy2210xxx , BestFit, Dynamic
import matplotlib.pyplot as plt
from analyze_result import Result, plot_comparison
import copy
import time
import pandas as pd
plt.ion()

results_df = pd.DataFrame(columns=[
    'Idx', 'Policy', 'Number of products', 'Average size of products', 'Number of stocks',
    'Average size of stocks', 'Product type', 'Max product per type',
    'Run time (s)', 'Number of stock used', 'Waste (%)'
])

records = []

def step(action, sum_products, sum_stocks):
        stock_idx = action["stock_idx"]
        size = action["size"]
        position = action["position"]

        width, height = size
        x, y = position

        # Check if the product is in the product list
        product_idx = None
        for i, product in enumerate(sum_products):
            if np.array_equal(product["size"], size):
                if product["quantity"] == 0:
                    continue

                product_idx = i  # Product index starts from 0
                break

        if product_idx is not None:
            if 0 <= stock_idx < len(sum_stocks):
                stock = sum_stocks[stock_idx]
                # Check if the product fits in the stock
                stock_width = np.sum(np.any(stock != -2, axis=1))
                stock_height = np.sum(np.any(stock != -2, axis=0))
                if (
                    x >= 0
                    and y >= 0
                    and x + width <= stock_width
                    and y + height <= stock_height
                ):
                    # Check if the position is empty
                    if np.all(stock[x : x + width, y : y + height] == -1):
                        stock[x : x + width, y : y + height] = product_idx
                        sum_products[product_idx]["quantity"] -= 1

        # An episode is done iff the all product quantities are 0
        terminated = all([product["quantity"] == 0 for product in sum_products])
        reward = 1 if terminated else 0  # Binary sparse rewards\

        observation = {"stocks": sum_stocks, "products": sum_products}
        info = 0

        return observation, reward, False, False, info

max_product_per_type = 25

count_run = 0

if __name__ == "__main__":
    print("Testing the policy")
    for num_stocks in range(10, 100, 10):
            max_product_type = 10
            # Reset the environment
            results_df = results_df.iloc[0:0]
            env = gym.make(
                "gym_cutting_stock/CuttingStock-v0",
                # render_mode="human",
                min_w=50,
                min_h=50,
                max_w=100,
                max_h=100,
                num_stocks=num_stocks,
                max_product_type=max_product_type,
                max_product_per_type=max_product_per_type,
            )
            observation, info = env.reset(seed=42)

            # Get the initial observation
            prods = observation["products"]
            prods_type = len(prods)
            sum_stocks = observation['stocks']
            NUM_EPISODES = np.sum([prod["quantity"] for prod in prods])
            average_size = np.sum([prod["size"][0] * prod["size"][1] * prod["quantity"] for prod in prods]) / NUM_EPISODES
            average_size_stock = np.sum([np.sum(np.any(stock != -2, axis=1) * np.sum(np.any(stock != -2, axis=0))) for stock in sum_stocks]) / num_stocks

            # Create a list of policies
            policy2210xxx = Policy2210xxx()
            result_policy2210xxx = Result(policy2210xxx, "Greedy Sorted", observation=copy.deepcopy(observation), count=0)
            greedy = GreedyPolicy()
            greedy_result = Result(greedy, "Greedy", observation=copy.deepcopy(observation), count=2)
            random_policy = RandomPolicy()
            random_result = Result(random_policy, "Random", observation=copy.deepcopy(observation), count=4)
            best_fit = BestFit()
            best_fit_result = Result(best_fit, "BestFit", observation=copy.deepcopy(observation), count=6)
            dynamic = Dynamic()
            dynamic_result = Result(dynamic, "Dynamic", observation=copy.deepcopy(observation), count=8)

            list_policies = {
                'BestFit': {
                    "policy": best_fit,
                    "result": best_fit_result,
                },
                'Policy2210xxx': {
                    "policy": policy2210xxx,
                    "result": result_policy2210xxx,
                },
                'Greedy': {
                    "policy": greedy,
                    "result": greedy_result,
                },
                'Random': {
                    "policy": random_policy,
                    "result": random_result,
                },
                'Dynamic': {
                    "policy": dynamic,
                    "result": dynamic_result,
                },
            }

            # Run the policies
            count_run += 1
            print(f'Run {count_run}: Number of stocks: {num_stocks}, Product type: {prods_type}, Max product per type: {max_product_per_type}')
            print('Number of products:', NUM_EPISODES)
            for _ in range(NUM_EPISODES):
                for policy in list_policies.values():
                    start_time = time.time()

                    action = policy["policy"].get_action(policy["result"].observation, info)
                    
                    execution_time = time.time() - start_time

                    policy["result"].time_execution.append(execution_time)

                    if action is not None:
                        sum_products = policy["result"].observation["products"]
                        sum_stocks = policy["result"].observation["stocks"]
                        policy["result"].observation, reward, terminated, truncated, info = step(action, sum_products, sum_stocks)

                        # Tính toán num_stock và num_empty cho Policy2210xxx
                        policy["result"].add_result(policy["result"].observation)

                        if terminated or truncated:
                            policy["result"].observation, info = env.reset()
                            break

                # Create a list of results
                list_all_result = [policy["result"] for policy in list_policies.values()]

                # Vẽ biểu đồ so sánh
                plot_comparison(list_all_result)

            # sum execution time of all policies
            for policy in list_policies.values():
                new_row = {
                    'Idx': count_run,
                    'Policy': policy["result"].policy_name,
                    'Number of products': NUM_EPISODES,
                    'Average size of products': average_size,
                    'Number of stocks': num_stocks,
                    'Average size of stocks': average_size_stock,
                    'Product type': prods_type,
                    'Max product per type': max_product_per_type,
                    'Run time (s)': np.sum(policy["result"].time_execution),
                    'Number of stock used': policy["result"].list_result[-1][0],
                    'Waste (%)': policy["result"].list_result[-1][1]
                }
                records.append(new_row)

            # Save the results to a csv file
            env.close()

# Save the results to a csv file
results_df = pd.DataFrame(records)
results_df.to_csv('results.csv', index=False)
    

    # test_policy = DQNCuttingStock()
    # for _ in range(NUM_EPISODES):
    #     with open ("log.txt", "a") as f:
    #         f.write(f'Observation: {observation}\n')
    #     action = test_policy.get_action(observation, info)
    #     print(action)
    #     observation, reward, terminated, truncated, info = env.step(action)
    #     if terminated or truncated:
    #         observation, info = env.reset()
    #         break

    # policy = ActorCriticCuttingStock()
    # for episode in range(NUM_EPISODES):
    #     action = policy.get_action(observation, info)
    #     observation, reward, terminated, truncated, info = env.step(action)
    #     if terminated or truncated:
    #         observation, info = env.reset()
    
