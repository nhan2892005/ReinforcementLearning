from policy import Policy
import numpy as np
import random

class Policy2210xxx(Policy):
    def __init__(self):
        super().__init__()
        self.name = "2210xxx"
        self.steps = []
        self.call_model = False
    
    def find_empty_region(self, stock, prod_size):
        rows, cols = stock.shape
        empty_regions = []
        prod_height, prod_width = prod_size
        
        # Find empty region in stock
        for i in range(rows - prod_height + 1):
            for j in range(cols - prod_width + 1):
                # Add space if can place product
                if np.all(stock[i:i+prod_height, j:j+prod_width] == -1):
                    empty_regions.append((i, j))
        
        return empty_regions

    def model(self, observation, info):
        # Get information of `products` and `stocks`
        products = observation["products"]
        sorted_products = sorted(products, key=lambda x: x['size'][0] * x['size'][1], reverse=True)
        stocks = observation["stocks"]
        stocks = np.array(stocks)

        for _, product in enumerate(sorted_products):
            prod_size = product['size']
            prod_quantity = product['quantity']
            if prod_quantity == 0:
                continue
            
            for stock_index, stock in enumerate(stocks):
                empty_regions = self.find_empty_region(stock, prod_size)

                if not empty_regions:
                    continue
                    
                selected_position = min(empty_regions, key=lambda pos: (pos[0], pos[1]))

                this_step = {
                    "stock_idx": stock_index,
                    "size": prod_size.tolist(),
                    "position": selected_position,
                } 

                return this_step

    def get_action(self, observation, info):
        step = self.model(observation, info)
        return step

    def place_product(self, stock, prod_size, position, product_id):
        i, j = position
        prod_height, prod_width = prod_size
        stock[i:i+prod_height, j:j+prod_width] = product_id 
        return stock
    
class GeneticsPolicy(Policy):
    def __init__(self):
        super().__init__()

    def find_empty_region(self, stock, prod_size):
        rows, cols = stock.shape
        empty_regions = []
        prod_height, prod_width = prod_size
        
        # Find empty region in stock
        for i in range(rows - prod_height + 1):
            for j in range(cols - prod_width + 1):
                # Add space if can place product
                if np.all(stock[i:i+prod_height, j:j+prod_width] == -1):
                    empty_regions.append((i, j))
        
        return empty_regions

    def can_increase_stock(self, stocks, idx_stock):
        num_stocks_used = sum([np.any(stock >= 0) for stock in stocks])
        return num_stocks_used < idx_stock
        
        
    def create_individual(self, stocks, products):
        '''
        Create individual (arrangement of products)
        @param `stocks`: list, list of stocks
        @param `products`: list, list of products
        * Return: individual (list)
        '''
        individual = []
        remaining_products = [product for product in products if product['quantity'] > 0]
        num_products = len(remaining_products)

        for idx_stock, stock in enumerate(stocks):
            for product in remaining_products:
                empty_regions = self.find_empty_region(stock, product['size'])
                # random select position in empty regions to place product
                i = j = 0
                if empty_regions:
                    i, j = random.choice(empty_regions)
                individual.append((idx_stock, i, j, product))

            if len(individual) == num_products:
                break

        return individual
    
    def population(self, stocks, products, population_size=20):
        '''
        Create population of individuals
        @param `stocks`: list, list of stocks
        @param `products`: list, list of products
        @param `population_size`: int, number of individuals
        * Return: population (list)
        '''
        population = []
        for _ in range(population_size):
            individual = self.create_individual(stocks, products)
            population.append(individual)
        return population

    def fitness(self, individual, stocks):
        '''
        Calculate fitness score of individual
        @param `individual`: list, arrangement of products
        @param `stocks`: list, list of stocks
        * Return: fitness_score (int)
        '''
        fitness_score = 0
        if not individual:
            return fitness_score
        stocks = np.array(stocks)
        for (idx_stock, i, j, prod) in individual:
            stock = stocks[idx_stock]
            if self._can_place_(stock, (i, j), prod['size']):
                fitness_score += 2
                if self.can_increase_stock(stocks, idx_stock):
                    fitness_score -= 1
                break
        return fitness_score

    def crossover(self, parents):
        '''
        Crossover between two parents
        @param `parent1`: list, arrangement of products
        @param `parent2`: list, arrangement of products
        * Return: child (list)
        '''
        child = []
        for parent in parents:
            # choose some best genes from parents
            if random.random() < 0.5:
                child.append(parent)

    def mutation(self, child, stocks, mutation_rate=0.3):
        '''
        Mutation for individual
        @param `individual`: list, arrangement of products
        @param `mutation_rate`: float, mutation rate
        * Return: individual (list)
        '''
        for i in range(len(child)):
            if random.random() < mutation_rate:
                weight, height = self._get_stock_size_(stocks[child[i][0]])
                h_i = random.randint(0, height - child[i][3]['size'][1])
                w_i = random.randint(0, weight - child[i][3]['size'][0])
                child[i] = (child[i][0], w_i, h_i, child[i][3])
    
    # * Function to get action from policy
    def get_action(self, observation, info):
        '''
        Get action from policy
        @param `observation`: dict, observation from environment
                              contains: list of products, list of stocks
        @param `info`: dict, extra information

        * Return: action (dict)
                  contains stock_idx: int, index of stock
                           size: list, size of product
                           position: tuple, position to place product
        '''
        # Get information of `products` and `stocks`
        products = observation["products"]
        stocks = observation["stocks"]
        
        # Create population
        population = self.population(stocks, products)

        # Evolutionary algorithm
        for idx in range(10):
            # Calculate fitness score
            fitness_scores = [self.fitness(individual, stocks) for individual in population]
            if sum(fitness_scores) == 0:
                break
            # Select parents
            parents = random.choices(population, weights=fitness_scores, k=10)
            # Crossover
            child = self.crossover(parents)
            # Mutation
            child = self.mutation(child, stocks)
            # Replace the worst individual
            worst_idx = fitness_scores.index(min(fitness_scores))
            population[worst_idx] = child

        # Get the best individual
        best_idx = fitness_scores.index(max(fitness_scores))
        best_individual = population[best_idx]
        best_fitness = fitness_scores[best_idx]

        # Get the first product that can be placed
        for (stock_idx, i, j, prod) in best_individual:
            stock = stocks[stock_idx]
            if self._can_place_(stock, (i, j), prod['size']):
                return {"stock_idx": stock_idx, "size": prod['size'], "position": (i, j)}
                
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
        
