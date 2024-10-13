# Idea

- Agent: Game, Model
- Training
  1. state = game.get_state()
  2. action = game.get_move(state)
  3. model.predict()
  4. reward, score, game_over = game.do_move(action)
  5. new_state = game.get_state()
  6. remember(state, action, reward, new_state, game_over)
  7. model.train()

- Game (Pygame)

- Model (Pytorch)

- Reward Policy:
    1. reward = -10 if game_over
    2. reward = 10 if score was increased
    3. else reward = 0

- Action Policy:
    [1, 0, 0] -> Straight
    [0, 1, 0] -> Right
    [0, 0, 1] -> Left

- State:
    [danger straight, danger right, danger left, 
     dir_left, dir_right, dir_up, dir_down,
     food_left, food_right, food_up, food_down]

- Model:
    1. Input: 11
    2. Hidden: 120
    3. Output: 3
    4. Using Deep Q-Learning
1. Bellman Equation
   New Q(s, a) = Q(s, a) + α * (R(s) + γ * max(Q(s', a')) - Q(s, a))
   Where:
     - Q(s, a): Q-value of state s and action a
     - α: Learning rate
     - R(s): Reward of state s
     - γ: Discount factor
     - max(Q(s', a')): Maximum Q-value of next state s' and all possible actions a'
2.  Q update rule simplified
    Q = model.predict(state)
    Q_new = R + γ * max(model.predict(new_state))

3. Loss function: MSE