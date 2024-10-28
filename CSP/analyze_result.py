import numpy as np
from policy import Policy
import matplotlib.pyplot as plt
from IPython import display
import random

count = 0
colors = ['b', 'r', 'g', 'y', 'm', 'c', 'k', 'brown']
markers = ['s', 'D', '^', 'v', 'p', '*']

class Result:
    def __init__(self, policy:Policy, name:str):
        global count
        self.list_result = []
        self.policy = policy
        self.policy_name = name
        self.stock_color = colors[count]
        self.product_color = colors[count + 1]
        count += 2

    def _get_num_stock_not_empty(self, stocks):
        return sum([np.any(stock >= 0) for stock in stocks])
    
    def _stock_is_empty(self, stock):
        return np.all(stock < 0)
    
    def _get_num_all_empty_regions(self, stocks):
        num_empty_regions = 0
        for stock in stocks:
            if self._stock_is_empty(stock):
                continue
            stock_w, stock_h = self.policy._get_stock_size_(stock)
            for x in range(stock_w):
                for y in range(stock_h):
                    if stock[x, y] == -1:
                        num_empty_regions += 1
        return num_empty_regions

    def add_result(self, observation):
        num_stock = self._get_num_stock_not_empty(observation["stocks"])
        num_empty = self._get_num_all_empty_regions(observation["stocks"])
        s_area = 0
        for stock in observation["stocks"]:
            w, h = self.policy._get_stock_size_(stock)
            s_area += w * h
        num_empty /= s_area
        num_empty *= 100
        print(f'Num Stock: {num_stock}, Num Empty: {num_empty}')
        self.list_result.append((num_stock, num_empty))

    def get_result(self):
        return self.list_result
    
def plot_comparison(
    list_all_result: list[Result]
):
    # Clear the current figure
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    
    # Create a new figure
    plt.title("Comparison of Policies")
    plt.xlabel("Episode")
    plt.ylabel("Values")

    for result in list_all_result:
        list_result = result.get_result()
        num_stock = [x[0] for x in list_result]
        num_empty = [x[1] for x in list_result]
        plt.bar(range(len(num_stock)), num_stock, label=f"{result.policy_name} - Num Stock", color=result.stock_color, alpha=0.5)
        plt.plot(range(len(num_empty)), num_empty, label=f"{result.policy_name} - Num Empty", color=result.product_color)

    plt.legend()
    plt.show(block=False)
    plt.savefig("comparison.png")
    plt.pause(0.1)