import numpy as np
from queue import PriorityQueue

# Sản phẩm và kích thước stock
products = [
    {'size': np.array([2, 5]), 'quantity': 1},  # Sản phẩm 1
    {'size': np.array([1, 3]), 'quantity': 2}   # Sản phẩm 2
]

stock_height = 5
stock_width = 20

# Hàm kiểm tra có thể đặt sản phẩm vào stock không
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

# Đặt sản phẩm vào stock
def place_product(stock, prod_size, pos):
    i, j = pos
    prod_height, prod_width = prod_size
    new_stock = stock.copy()
    for x in range(i, i + prod_height):
        for y in range(j, j + prod_width):
            new_stock[x][y] = 0  # Đánh dấu vị trí đã đặt sản phẩm
    return new_stock

# Node của cây Branch and Bound
class Node:
    def __init__(self, level, stock, placed_items, bound):
        self.level = level            # Cấp độ của node trong cây
        self.stock = stock            # Trạng thái stock hiện tại
        self.placed_items = placed_items  # Danh sách các sản phẩm đã đặt
        self.bound = bound            # Giá trị bound (đánh giá khả năng tối ưu của nhánh)

    # So sánh dựa trên bound để chọn nhánh tốt nhất trước
    def __lt__(self, other):
        return self.bound > other.bound

# Hàm tính bound cho một node
def calculate_bound(node, products):
    remaining_area = np.sum(node.stock == -1)  # Diện tích trống còn lại
    placed_area = sum(np.prod(product['size']) for product in node.placed_items)  # Diện tích đã đặt
    total_area = remaining_area + placed_area

    return total_area

# Thuật toán Branch and Bound để xếp sản phẩm
def branch_and_bound(stock_height, stock_width, products):
    initial_stock = np.full((stock_height, stock_width), -1)  # Stock ban đầu trống
    root = Node(0, initial_stock, [], 0)
    
    best_node = None
    best_fitness = 0
    queue = PriorityQueue()
    queue.put(root)

    while not queue.empty():
        node = queue.get()

        # Nếu tất cả các sản phẩm đã được đặt
        if node.level == len(products):
            fitness = len(node.placed_items)
            if fitness > best_fitness:
                best_fitness = fitness
                best_node = node
            continue

        # Lấy sản phẩm hiện tại cần đặt
        product = products[node.level]

        # Tìm tất cả các vị trí có thể đặt sản phẩm này vào stock
        for i in range(stock_height):
            for j in range(stock_width):
                if can_place(node.stock, product['size'], (i, j)):
                    new_stock = place_product(node.stock, product['size'], (i, j))
                    new_placed_items = node.placed_items + [product]
                    new_bound = calculate_bound(Node(node.level + 1, new_stock, new_placed_items, 0), products)

                    # Nếu bound lớn hơn fitness tốt nhất hiện tại, tiếp tục phát triển nhánh này
                    if new_bound > best_fitness:
                        new_node = Node(node.level + 1, new_stock, new_placed_items, new_bound)
                        queue.put(new_node)

    return best_node

# Chạy thuật toán Branch and Bound
best_solution = branch_and_bound(stock_height, stock_width, products)

# In ra kết quả tốt nhất
if best_solution:
    print("Best solution found with fitness:", len(best_solution.placed_items))
    print("Placed items:", best_solution.placed_items)
else:
    print("No solution found.")
