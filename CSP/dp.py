import numpy as np

def can_place(stock, dp, prod_size, pos):
    """Kiểm tra xem có thể đặt sản phẩm tại vị trí pos hay không"""
    i, j = pos
    prod_height, prod_width = prod_size
    stock_height, stock_width = stock.shape
    
    # Kiểm tra nếu sản phẩm vượt quá biên giới của stock
    if i + prod_height > stock_height or j + prod_width > stock_width:
        return False
    
    # Kiểm tra nếu vùng này đã được sử dụng trong dp
    for x in range(i, i + prod_height):
        for y in range(j, j + prod_width):
            if dp[x][y] != -1:
                return False
    return True

def place_product(dp, prod_size, pos):
    """Đặt sản phẩm vào bảng dp"""
    i, j = pos
    prod_height, prod_width = prod_size
    for x in range(i, i + prod_height):
        for y in range(j, j + prod_width):
            dp[x][y] = 0  # Giả sử 0 là mã của sản phẩm
    return dp

def dynamic_programming_placement(stock, products):
    stock_height, stock_width = stock.shape
    dp = np.full_like(stock, -1)  # Bảng DP với -1 đại diện cho ô trống
    
    for product in products:
        prod_size = product['size']
        quantity = product['quantity']
        
        for _ in range(quantity):
            placed = False
            # Duyệt qua từng ô của stock để tìm vị trí đặt
            for i in range(stock_height):
                for j in range(stock_width):
                    if can_place(stock, dp, prod_size, (i, j)):
                        # Nếu đặt được, cập nhật dp và đánh dấu đã đặt
                        dp = place_product(dp, prod_size, (i, j))
                        placed = True
                        print({"stock_idx": (i, j), "size": prod_size, "position": (i, j)})
                        break
                if placed:
                    break
            
            if not placed:
                print("No suitable region found for product size", prod_size)
    
    return dp

# Dữ liệu ví dụ
stock = np.array([[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, -2, -2],
                  [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, -2, -2],
                  [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, -2, -2],
                  [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, -2, -2],
                  [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, -2, -2]])

products = [
    {'size': np.array([2, 5]), 'quantity': 1},  # Sản phẩm 1
    {'size': np.array([1, 3]), 'quantity': 2}   # Sản phẩm 2
]

# Gọi thuật toán Dynamic Programming
dp = dynamic_programming_placement(stock, products)

print("Updated stock (DP table):")
print(dp)
