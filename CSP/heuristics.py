import numpy as np

def find_empty_region(stock, prod_size):
    rows, cols = stock.shape
    empty_regions = []
    prod_height, prod_width = prod_size
    
    # Duyệt qua từng ô của stock để tìm vị trí bắt đầu của vùng trống
    for i in range(rows - prod_height + 1):
        for j in range(cols - prod_width + 1):
            # Kiểm tra nếu vùng từ (i, j) có thể đặt được sản phẩm
            if np.all(stock[i:i+prod_height, j:j+prod_width] == -1):
                empty_regions.append((i, j))
    
    return empty_regions

def place_product(stock, prod_size, position, product_id):
    i, j = position
    prod_height, prod_width = prod_size
    stock[i:i+prod_height, j:j+prod_width] = product_id 
    return stock

def heuristic_placement(stocks, products):
    for prod_idx, product in enumerate(products):
        prod_size = product['size']
        quantity = product['quantity']
        
        for _ in range(quantity):
            placed = False  # Biến đánh dấu xem sản phẩm đã được đặt chưa
            for stock_index, stock in enumerate(stocks):
                empty_regions = find_empty_region(stock, prod_size)
                
                if not empty_regions:
                    continue
                
                # Tiêu chí chọn vị trí: chọn vị trí gần góc trên bên trái nhất
                selected_position = min(empty_regions, key=lambda pos: (pos[0], pos[1]))
                
                # Đặt sản phẩm vào stock
                stocks[stock_index] = place_product(stock, prod_size, selected_position, prod_idx)
                
                # Trả về thông tin vị trí đặt sản phẩm
                print({
                    "stock_idx": stock_index,
                    "size": prod_size.tolist(),
                    "position": selected_position,
                    'product_id': prod_idx
                })
                
                placed = True  # Đánh dấu sản phẩm đã được đặt
                break  # Thoát khỏi vòng lặp stock khi đã đặt thành công
            
            if not placed:
                print(f"No suitable region found for product size {prod_size.tolist()}")

    return stocks

# Dữ liệu ví dụ cho nhiều stock
stocks = [
    np.array([[ 0,  0, -1, -1, -1, -2],
              [ 0,  0, -1, -1, -1, -2],
              [-1, -1, -1, -1, -1, -2]]),  # Stock 1
    np.array([[-1, -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1, -1]]),  # Stock 2
    np.array([[-2, -1, -1, -1, -1, -1],
              [-2, -1, -1, -1, -1, -1],
              [-2, -2, -2, -2, -2, -2]]),   # Stock 3
    np.array([[-2, -2, -2, -2, -2, -2],
              [-1, -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1, -1]])   # Stock 4
]

products = [
    {'size': np.array([2, 2]), 'quantity': 1},   # Sản phẩm 0
    {'size': np.array([1, 3]), 'quantity': 2},   # Sản phẩm 1
    {'size': np.array([2, 1]), 'quantity': 1},   # Sản phẩm 2
    {'size': np.array([1, 1]), 'quantity': 6},   # Sản phẩm 3
    {'size': np.array([2, 1]), 'quantity': 1},   # Sản phẩm 4
]

# Gọi thuật toán heuristics
updated_stocks = heuristic_placement(stocks, products)

print("Updated stocks:")
for i, stock in enumerate(updated_stocks):
    print(f"Stock {i + 1}:")
    print(stock)
