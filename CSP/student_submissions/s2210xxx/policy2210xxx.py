from policy import Policy
import numpy as np

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