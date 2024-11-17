from pulp import *

n = 8
weights =    [ 4,  2,  8,  3,  7,  6,  9,  5]
prices =     [19, 17, 30, 13, 25, 10, 23, 29]
#*decision = [ 0,  1,  0,  1,  1,  0,  0,  1]
carry_weight = 17

model = LpProblem("Knapsack", sense=LpMaximize)

x = [LpVariable(f"x_{i + 1}", cat=LpBinary) for i in range(n)]

model += lpDot(prices, x)
model += lpDot(weights, x) <= carry_weight

status = model.solve()

print("Optimal value:", value(model.objective))
print("Optimal solution:")
for idx, v in enumerate(model.variables()):
    if v.varValue:
        print(f"  - {v.name} = {v.varValue} -> weight = {weights[idx]}, price = {prices[idx]}")
