import numpy as np
import random

# Khởi tạo stock và sản phẩm
stock_height = 5
stock_width = 20
products = [
    {'size': np.array([2, 5]), 'quantity': 1},  # Sản phẩm 1
    {'size': np.array([1, 3]), 'quantity': 2}   # Sản phẩm 2
]

# Khởi tạo một cá thể (bố trí sản phẩm)
def create_individual(stock_height, stock_width, products):
    individual = []
    for product in products:
        for _ in range(product['quantity']):
            # Tạo vị trí ngẫu nhiên trong stock
            i = random.randint(0, stock_height - product['size'][0])
            j = random.randint(0, stock_width - product['size'][1])
            individual.append((i, j, product['size']))
    return individual

# Hàm fitness đánh giá cá thể
def fitness(individual, stock_height, stock_width):
    stock = np.full((stock_height, stock_width), -1)  # Stock ban đầu trống
    fitness_score = 0

    # Đặt sản phẩm vào stock nếu có thể
    for (i, j, size) in individual:
        if can_place(stock, size, (i, j)):
            stock = place_product(stock, size, (i, j))
            fitness_score += 1  # Cộng điểm nếu đặt thành công

    return fitness_score

# Hàm kiểm tra và đặt sản phẩm (cùng như DP)
def can_place(stock, prod_size, pos):
    i, j = pos
    prod_height, prod_width = prod_size
    stock_height, stock_width = stock.shape
    
    if i + prod_height > stock_height or j + prod_width > stock_width:
        return False
    
    for x in range(i, i + prod_height):
        for y in range(j, j + prod_width):
            if stock[x][y] != -1:
                return False
    return True

def place_product(stock, prod_size, pos):
    i, j = pos
    prod_height, prod_width = prod_size
    for x in range(i, i + prod_height):
        for y in range(j, j + prod_width):
            stock[x][y] = 0  # Giả sử 0 là mã sản phẩm
    return stock

# Hàm lai ghép (crossover)
def crossover(parent1, parent2):
    child = []
    split = len(parent1) // 2
    child.extend(parent1[:split])
    child.extend(parent2[split:])
    return child

# Hàm đột biến (mutation)
def mutate(individual, mutation_rate=0.01):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            i = random.randint(0, stock_height - individual[i][2][0])
            j = random.randint(0, stock_width - individual[i][2][1])
            individual[i] = (i, j, individual[i][2])  # Thay đổi vị trí
    return individual

# Chọn lọc tự nhiên (select individuals based on fitness)
def select(population, fitnesses, num_individuals):
    selected = random.choices(population, weights=fitnesses, k=num_individuals)
    return selected

# Khởi tạo quần thể (population)
def create_population(pop_size, stock_height, stock_width, products):
    return [create_individual(stock_height, stock_width, products) for _ in range(pop_size)]

# Chạy thuật toán di truyền
def genetic_algorithm(stock_height, stock_width, products, generations=100, pop_size=20, mutation_rate=0.01):
    population = create_population(pop_size, stock_height, stock_width, products)
    
    for generation in range(generations):
        # Tính toán fitness của từng cá thể trong quần thể
        fitnesses = [fitness(ind, stock_height, stock_width) for ind in population]
        print(f"Generation {generation}, best fitness: {max(fitnesses)}")
        
        # Chọn lọc các cá thể tốt nhất để lai ghép
        selected = select(population, fitnesses, pop_size // 2)
        
        # Tạo thế hệ mới bằng cách lai ghép
        new_population = []
        for i in range(pop_size // 2):
            parent1 = random.choice(selected)
            parent2 = random.choice(selected)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            new_population.append(child)
        
        population = new_population
    
    # Trả về cá thể tốt nhất sau khi tiến hóa
    best_individual = max(population, key=lambda ind: fitness(ind, stock_height, stock_width))
    return best_individual

# Gọi thuật toán di truyền
best_layout = genetic_algorithm(stock_height, stock_width, products)

print("Best layout:")
print(best_layout)
