import gym_cutting_stock
import gymnasium as gym
import numpy as np
from policy import GreedyPolicy, RandomPolicy
from student_submissions.s2210xxx.policy2210xxx import Policy2210xxx, GeneticsPolicy
import matplotlib.pyplot as plt
from analyze_result import Result, plot_comparison
plt.ion()

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
        reward = 1 if terminated else 0  # Binary sparse rewards

        observation = {"stocks": sum_stocks, "products": sum_products}
        info = 0

        return observation, reward, False, False, info

if __name__ == "__main__":
    print("Testing the policy")
    
    # Reset the environment
    env = gym.make(
        "gym_cutting_stock/CuttingStock-v0",
        # min_w=10,
        # max_w=20,
        # min_h=10,
        # max_h=20,
    )
    
    NUM_EPISODES = 1000
    observation, info = env.reset(seed=42)

    sum_products = observation['products']
    sum_stocks = observation['stocks']
    backup_products = sum_products
    backup_stocks = sum_stocks
    observation_2210xxx = observation
    observation_greedy = observation

    policy2210xxx = Policy2210xxx()
    result_policy2210xxx = Result(policy2210xxx, "Policy2210xxx")
    greedy = GreedyPolicy()
    greedy_result = Result(greedy, "Greedy")
    random_policy = RandomPolicy()
    random_result = Result(random_policy, "Random")
    genetic = GeneticsPolicy()
    genetic_result = Result(genetic, "Genetic")

    list_policies = {
        'Policy2210xxx': {
            "policy": policy2210xxx,
            "result": result_policy2210xxx,
            "observation": observation
        },
        'Greedy': {
            "policy": greedy,
            "result": greedy_result,
            "observation": observation
        },
        'Random': {
            "policy": random_policy,
            "result": random_result,
            "observation": observation
        },
        'Genetic': {
            "policy": genetic,
            "result": genetic_result,
            "observation": observation
        },
    }

    prods = observation["products"]
    sum_prods = sum([prod["quantity"] for prod in prods])
    average_size = sum([prod["size"][0] * prod["size"][1] for prod in prods]) / sum_prods

    for episode in range(NUM_EPISODES):
        for policy in list_policies.values():
            action = policy["policy"].get_action(policy["observation"], info)
            if action is not None:
                policy["observation"], reward, terminated, truncated, info = step(action, sum_products, sum_stocks)

                # Tính toán num_stock và num_empty cho Policy2210xxx
                policy["result"].add_result(policy["observation"])

                if terminated or truncated:
                    policy["observation"], info = env.reset()
                    break

        # Create a list of results
        list_all_result = [result_policy2210xxx, greedy_result, random_result, genetic_result]

        # Vẽ biểu đồ so sánh
        plot_comparison(list_all_result)

        num_prod = sum([prod["quantity"] for prod in observation_greedy["products"]])
        if num_prod == 0:
            break

    # genetic = GeneticsPolicy()
    # for _ in range(NUM_EPISODES):
    #     action = genetic.get_action(observation, info)
    #     observation, reward, terminated, truncated, info = env.step(action)
    #     if terminated or truncated:
    #         observation, info = env.reset()
    #         break
    env.close()
