import pulp

# Thông tin về sản phẩm (chiều dài, chiều rộng, số lượng)
products = [
    {'size': (2, 3), 'quantity': 3},  # Sản phẩm 1
    {'size': (1, 5), 'quantity': 2},  # Sản phẩm 2
    {'size': (4, 2), 'quantity': 1},  # Sản phẩm 3
]

# Thông tin về các stock (chiều dài, chiều rộng)
stocks = [
    {'size': (10, 10)},  # Stock 1
    {'size': (15, 5)},   # Stock 2
]

# Khởi tạo bài toán tối ưu hóa
problem = pulp.LpProblem("2D Cutting Stock Problem", pulp.LpMinimize)

# Biến nhị phân x[i][j][k][l] đại diện cho việc có đặt sản phẩm i vào stock j tại vị trí (k, l) hay không
x = {}
for i, product in enumerate(products):
    for j, stock in enumerate(stocks):
        for k in range(stock['size'][0] - product['size'][0] + 1):
            for l in range(stock['size'][1] - product['size'][1] + 1):
                x[(i, j, k, l)] = pulp.LpVariable(f"x_{i}_{j}_{k}_{l}", cat="Binary")

# Hàm mục tiêu: Tối thiểu hóa tổng diện tích stock sử dụng
problem += pulp.lpSum(x[(i, j, k, l)] * stock['size'][0] * stock['size'][1]
                      for i, product in enumerate(products)
                      for j, stock in enumerate(stocks)
                      for k in range(stock['size'][0] - product['size'][0] + 1)
                      for l in range(stock['size'][1] - product['size'][1] + 1)), "Minimize Stock Usage"

# Ràng buộc 1: Mỗi sản phẩm phải được đặt đúng số lượng
for i, product in enumerate(products):
    problem += pulp.lpSum(x[(i, j, k, l)] for j in range(len(stocks))
                          for k in range(stocks[j]['size'][0] - product['size'][0] + 1)
                          for l in range(stocks[j]['size'][1] - product['size'][1] + 1)) == product['quantity'], f"Product_{i}_Quantity"

# Ràng buộc 2: Không chồng chéo sản phẩm
for j, stock in enumerate(stocks):
    grid = [[0] * stock['size'][1] for _ in range(stock['size'][0])]
    for i, product in enumerate(products):
        for k in range(stock['size'][0] - product['size'][0] + 1):
            for l in range(stock['size'][1] - product['size'][1] + 1):
                for dx in range(product['size'][0]):
                    for dy in range(product['size'][1]):
                        grid[k + dx][l + dy] += x[(i, j, k, l)]
                # Đảm bảo không vượt quá 1 sản phẩm ở mỗi vị trí
                problem += grid[k][l] <= 1, f"Non_Overlap_{i}_{j}_{k}_{l}"

# Ràng buộc 3: Sản phẩm không được vượt quá giới hạn của stock
for i, product in enumerate(products):
    for j, stock in enumerate(stocks):
        for k in range(stock['size'][0] - product['size'][0] + 1):
            for l in range(stock['size'][1] - product['size'][1] + 1):
                # Đảm bảo sản phẩm nằm trong giới hạn của stock
                if k + product['size'][0] > stock['size'][0] or l + product['size'][1] > stock['size'][1]:
                    problem += x[(i, j, k, l)] == 0, f"Out_Of_Bounds_{i}_{j}_{k}_{l}"

# Giải bài toán ILP
problem.solve()

# In kết quả
print("Trạng thái giải bài toán:", pulp.LpStatus[problem.status])

# In ra các sản phẩm được đặt ở đâu
print("Kết quả đặt sản phẩm:")
for i, product in enumerate(products):
    for j, stock in enumerate(stocks):
        for k in range(stock['size'][0] - product['size'][0] + 1):
            for l in range(stock['size'][1] - product['size'][1] + 1):
                if pulp.value(x[(i, j, k, l)]) == 1:
                    print(f"Sản phẩm {i+1} được đặt trong stock {j+1} tại vị trí ({k}, {l})")
