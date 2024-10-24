import numpy as np
import random

# Khởi tạo môi trường với các stock và sản phẩm
class CuttingStockEnv:
    def __init__(self, stock_size, products):
        self.stock_size = stock_size  # Kích thước stock (chiều dài, chiều rộng)
        self.products = products  # Danh sách sản phẩm (chiều dài, chiều rộng, số lượng)
        self.reset()

    def reset(self):
        # Reset trạng thái môi trường: khởi tạo một stock trống
        self.state = np.zeros(self.stock_size, dtype=int)  # Stock chưa có sản phẩm nào
        self.remaining_products = {i: p['quantity'] for i, p in enumerate(self.products)}  # Còn bao nhiêu sản phẩm cần cắt
        return self.state

    def step(self, action):
        # Action: Chọn một sản phẩm và đặt nó vào stock tại một vị trí
        product_idx, x, y = action
        product = self.products[product_idx]['size']
        
        # Kiểm tra xem sản phẩm có vừa vào vị trí này hay không
        if x + product[0] > self.stock_size[0] or y + product[1] > self.stock_size[1]:
            # Vượt quá kích thước stock -> hành động không hợp lệ
            return self.state, -10, False

        # Kiểm tra xem vị trí này có bị chiếm chưa
        if np.any(self.state[x:x + product[0], y:y + product[1]] > 0):
            # Vị trí đã bị chiếm -> hành động không hợp lệ
            return self.state, -10, False

        # Nếu hợp lệ, đặt sản phẩm vào vị trí này
        self.state[x:x + product[0], y:y + product[1]] = 1  # Đặt sản phẩm vào stock
        self.remaining_products[product_idx] -= 1  # Giảm số lượng sản phẩm cần đặt

        # Phần thưởng dựa trên diện tích sản phẩm
        reward = product[0] * product[1]

        # Kiểm tra xem có còn sản phẩm nào cần cắt không
        done = all(quantity == 0 for quantity in self.remaining_products.values())
        return self.state, reward, done

    def available_actions(self):
        # Lấy tất cả các hành động hợp lệ: chọn một sản phẩm và một vị trí trong stock
        actions = []
        for product_idx, product in enumerate(self.products):
            if self.remaining_products[product_idx] > 0:
                product_size = product['size']
                for x in range(self.stock_size[0] - product_size[0] + 1):
                    for y in range(self.stock_size[1] - product_size[1] + 1):
                        actions.append((product_idx, x, y))
        return actions

class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.95, epsilon=0.1):
        self.env = env
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.q_table = {}

    def get_q_value(self, state, action):
        # Trả về giá trị Q cho một trạng thái và hành động
        return self.q_table.get((tuple(state.flatten()), action), 0)

    def update_q_value(self, state, action, reward, next_state, done):
        # Tính toán giá trị Q cập nhật
        current_q = self.get_q_value(state, action)
        max_next_q = max(self.get_q_value(next_state, a) for a in self.env.available_actions())
        target = reward + (0 if done else self.gamma * max_next_q)
        self.q_table[(tuple(state.flatten()), action)] = (1 - self.alpha) * current_q + self.alpha * target

    def choose_action(self, state):
        # Chọn hành động: khám phá hoặc khai thác
        if random.uniform(0, 1) < self.epsilon:
            # Khám phá: chọn ngẫu nhiên
            return random.choice(self.env.available_actions())
        else:
            # Khai thác: chọn hành động có giá trị Q lớn nhất
            q_values = [(self.get_q_value(state, action), action) for action in self.env.available_actions()]
            max_q = max(q_values, key=lambda x: x[0])[0]
            best_actions = [action for q, action in q_values if q == max_q]
            return random.choice(best_actions)

# Khởi tạo môi trường và agent
stock_size = (10, 10)
products = [
    {'size': (2, 3), 'quantity': 3},
    {'size': (1, 5), 'quantity': 2},
    {'size': (4, 2), 'quantity': 1},
]

env = CuttingStockEnv(stock_size, products)
agent = QLearningAgent(env)

# Training agent
for episode in range(1000):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.update_q_value(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
    print(f"Episode {episode + 1}: Total reward = {total_reward}")
