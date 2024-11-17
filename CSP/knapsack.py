import gym_cutting_stock
import gymnasium as gym
import numpy as np
import pulp

env = gym.make(
                "gym_cutting_stock/CuttingStock-v0",
                # render_mode="human",
                min_w=10,
                min_h=20,
                max_w=10,
                max_h=20,
                num_stocks=5,
                max_product_type=10,
                max_product_per_type=3,
            )
observation, info = env.reset(seed=42)
env.close()

def _get_stock_size_(stock):
    stock_w = np.sum(np.any(stock != -2, axis=1))
    stock_h = np.sum(np.any(stock != -2, axis=0))

    return stock_w, stock_h

# Information of products (size, quantity)
products = observation['products']

stocks = []
# Get information of stocks (size)
for stock in observation['stocks']:
    stock_w, stock_h = _get_stock_size_(stock)
    stocks.append({'size': (stock_w, stock_h)})

print("Products:", products)
print("Stocks:", stocks)

# Initialize the problem
problem = pulp.LpProblem("2D Cutting Stock Problem", pulp.LpMinimize)

# Decision variables: x[i, j, k, l] = 1 if product i is placed in stock j at position (k, l)
x = {}
for i, product in enumerate(products):
    for j, stock in enumerate(stocks):
        for k in range(stock['size'][0] - product['size'][0] + 1):
            for l in range(stock['size'][1] - product['size'][1] + 1):
                x[(i, j, k, l)] = pulp.LpVariable(f"x_{i}_{j}_{k}_{l}", cat="Binary")

# Objective function: Minimize the stock usage
problem += pulp.lpSum(x[(i, j, k, l)] * stock['size'][0] * stock['size'][1]
                      for i, product in enumerate(products)
                      for j, stock in enumerate(stocks)
                      for k in range(stock['size'][0] - product['size'][0] + 1)
                      for l in range(stock['size'][1] - product['size'][1] + 1)), "Minimize Stock Usage"

# Constraint 1: Each product must be placed exactly once
for i, product in enumerate(products):
    problem += pulp.lpSum(x[(i, j, k, l)] for j in range(len(stocks))
                          for k in range(stocks[j]['size'][0] - product['size'][0] + 1)
                          for l in range(stocks[j]['size'][1] - product['size'][1] + 1)) == product['quantity'], f"Product_{i}_Quantity"

# Constraint 2: Non-overlapping constraint
for j, stock in enumerate(stocks):
    grid = [[0] * stock['size'][1] for _ in range(stock['size'][0])]
    for i, product in enumerate(products):
        for k in range(stock['size'][0] - product['size'][0] + 1):
            for l in range(stock['size'][1] - product['size'][1] + 1):
                for dx in range(product['size'][0]):
                    for dy in range(product['size'][1]):
                        grid[k + dx][l + dy] += x[(i, j, k, l)]
                # Ensure that the sum of all cells is at most 1
                problem += grid[k][l] <= 1, f"Non_Overlap_{i}_{j}_{k}_{l}"

# Constraint 3: Out of bounds constraint
for i, product in enumerate(products):
    for j, stock in enumerate(stocks):
        for k in range(stock['size'][0] - product['size'][0] + 1):
            for l in range(stock['size'][1] - product['size'][1] + 1):
                # Đảm bảo sản phẩm nằm trong giới hạn của stock
                if k + product['size'][0] > stock['size'][0] or l + product['size'][1] > stock['size'][1]:
                    problem += x[(i, j, k, l)] == 0, f"Out_Of_Bounds_{i}_{j}_{k}_{l}"

# Solve the problem
problem.solve()

print("Trạng thái giải bài toán:", pulp.LpStatus[problem.status])
# Print the result
print("Kết quả đặt sản phẩm:")
for i, product in enumerate(products):
    for j, stock in enumerate(stocks):
        for k in range(stock['size'][0] - product['size'][0] + 1):
            for l in range(stock['size'][1] - product['size'][1] + 1):
                if pulp.value(x[(i, j, k, l)]) == 1:
                    print(f"Sản phẩm {i+1} được đặt trong stock {j+1} tại vị trí ({k}, {l})")
