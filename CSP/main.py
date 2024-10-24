import gym_cutting_stock
import gymnasium as gym
from policy import GreedyPolicy, RandomPolicy
from student_submissions.s2210xxx.policy2210xxx import Policy2210xxx

# Create the environment
env = gym.make(
    "gym_cutting_stock/CuttingStock-v0",
    render_mode="human",
    min_w = 30,
    max_w = 40,
    min_h = 20,
    max_h = 40,
    max_product_type = 25,
    max_product_per_type = 20,
)
NUM_EPISODES = 1000

if __name__ == "__main__":
    print("Testing the policy")
    # Reset the environment
    # observation, info = env.reset(seed=42)

    # Test GreedyPolicy
    # gd_policy = GreedyPolicy()
    # ep = 0
    # while ep < NUM_EPISODES:
    #     action = gd_policy.get_action(observation, info)
    #     observation, reward, terminated, truncated, info = env.step(action)

    #     if terminated or truncated:
    #         observation, info = env.reset(seed=ep)
    #         print(info)
    #         ep += 1

    # Reset the environment
    observation, info = env.reset(seed=42)

    # Test RandomPolicy
    # rd_policy = RandomPolicy()
    # ep = 0
    # while ep < NUM_EPISODES:
        # action = rd_policy.get_action(observation, info)
        # observation, reward, terminated, truncated, info = env.step(action)

        # if terminated or truncated:
            # observation, info = env.reset(seed=ep)
            # print(info)
            # ep += 1

    # Uncomment the following code to test your policy
    # # Reset the environment
    # observation, info = env.reset(seed=42)
    with open("output.txt", "w") as f:
            f.write(f"Start Game with: {observation['products']}\n")
    policy2210xxx = Policy2210xxx()
    for _ in range(NUM_EPISODES):
        action = policy2210xxx.get_action(observation, info)
        with open("output.txt", "a") as f:
            f.write(f"{action}\n")
        # print to file
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()
            with open("output.txt", "a") as f:
                f.write(f"New Game with: {observation['products']}\n")

env.close()
