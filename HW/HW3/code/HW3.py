import numpy as np


class Race():
    def __init__(self, width, height, start, goal, obstacles):
        self.width = width
        self.height = height
        self.start = start
        self.goal = goal
        self.obstacles = obstacles

    def reset(self):
        self.state = self.start

    def move(self, action):
        acceleration = action.split("_")
        if acceleration[0] == "up" and self.state[1] < self.height - 1 and (
        self.state[0], self.state[1] + 1) not in self.obstacles:
            self.state = (self.state[0], self.state[1] + int(acceleration[1]))
        elif acceleration[0] == "right" and self.state[0] < self.width - 1 and (
        self.state[0] + 1, self.state[1]) not in self.obstacles:
            self.state = (self.state[0] + int(acceleration[1]), self.state[1])


class MonteCarloAccelerationRL:
    def __init__(self, maze, epsilon=0.1, gamma=0.9):
        self.maze = maze
        self.epsilon = epsilon
        self.gamma = gamma
        self.q_values = {}
        self.visits = {}

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            # Exploration: Choose a random acceleration action
            return np.random.choice(["up_-1", "up_0", "up_1", "right_-1", "right_0", "right_1"])
        else:
            # Exploitation: Choose the best acceleration action
            return max(self.q_values[state], key=self.q_values[state].get)

    def get_q_value(self, state, action):
        if state not in self.q_values:
            self.q_values[state] = {
                "up_-1": 0,
                "up_0": 0,
                "up_1": 0,
                "right_-1": 0,
                "right_0": 0,
                "right_1": 0
            }
            self.visits[state] = {
                "up_-1": 0,
                "up_0": 0,
                "up_1": 0,
                "right_-1": 0,
                "right_0": 0,
                "right_1": 0
            }
        return self.q_values[state][action]

    def update_q_values(self, episode):
        total_reward = 0
        for state, action, reward in reversed(episode):
            total_reward = self.gamma * total_reward + reward
            self.q_values[state][action] += total_reward
            self.visits[state][action] += 1

    def run_episode(self):
        episode = []
        self.maze.reset()
        while not self.maze.is_goal_reached():
            state = self.maze.state
            action = self.get_action(state)
            self.maze.move(action)
            reward = -1  # Constant reward for each step
            episode.append((state, action, reward))
        return episode

    def train(self, num_episodes):
        for _ in range(num_episodes):
            episode = self.run_episode()
            self.update_q_values(episode)

    def print_policy(self):
        for y in reversed(range(self.maze.height)):
            for x in range(self.maze.width):
                state = (x, y)
                if state in self.maze.obstacles:
                    print("X", end="\t")
                elif state == self.maze.goal:
                    print("G", end="\t")
                else:
                    best_action = max(self.get_q_value(state), key=self.get_q_value(state).get)
                    print(best_action.split("_")[0][0].upper() + best_action.split("_")[1], end="\t")
            print()


# Define maze parameters
width = 32
height = 32

grid = np.zeros((32,32))

grid[31, :] = [-1]*16 + [0]*(16-1) + [1]*1
grid[30, :] = [-1]*13 + [0]*(19-1) + [1]*1
grid[29, :] = [-1]*12 + [0]*(20-1) + [1]*1
grid[28, :] = [-1]*11 + [0]*(21-1) + [1]*1
grid[27, :] = [-1]*11 + [0]*(21-1) + [1]*1
grid[26, :] = [-1]*11 + [0]*(21-1) + [1]*1
grid[25, :] = [-1]*11 + [0]*(21-1) + [1]*1
grid[24, :] = [-1]*12 + [0]*(20-1) + [1]*1
grid[23, :] = [-1]*13 + [0]*(19-1) + [1]*1
grid[22, :] = [-1]*14 + [0]*(18-2) + [-1]*2
grid[21, :] = [-1]*14 + [0]*(18-5) + [-1]*5
grid[20, :] = [-1]*14 + [0]*(18-6) + [-1]*6
grid[19, :] = [-1]*14 + [0]*(18-8) + [-1]*8
grid[18, :] = [-1]*14 + [0]*(18-9) + [-1]*9
grid[17, :] = [-1]*13 + [0]*(19-9) + [-1]*9
grid[16, :] = [-1]*12 + [0]*(20-9) + [-1]*9
grid[15, :] = [-1]*11 + [0]*(21-9) + [-1]*9
grid[14, :] = [-1]*10 + [0]*(22-9) + [-1]*9
grid[13, :] = [-1]*9 + [0]*(23-9) + [-1]*9
grid[12, :] = [-1]*8 + [0]*(24-9) + [-1]*9
grid[11, :] = [-1]*7 + [0]*(25-9) + [-1]*9
grid[10, :] = [-1]*6 + [0]*(26-9) + [-1]*9
grid[9, :] = [-1]*5 + [0]*(27-9) + [-1]*9
grid[8, :] = [-1]*4 + [0]*(28-9) + [-1]*9
grid[7, :] = [-1]*3 + [0]*(29-9) + [-1]*9
grid[6, :] = [-1]*2 + [0]*(30-9) + [-1]*9
grid[5, :] = [-1]*1 + [0]*(31-9) + [-1]*9
grid[4, :] = [0]*(32-9) + [-1]*9
grid[3, :] = [0]*(32-9) + [-1]*9
grid[2, :] = [0]*(32-9) + [-1]*9
grid[1, :] = [0]*(32-9) + [-1]*9
grid[0, :] = [0]*(32-9) + [-1]*9

obstacles = []
# Iterate through the grid and identify obstacle positions
for i in range(len(grid)):
    for j in range(len(grid[i])):
        if grid[i, j] == -1:
            obstacles.append((i, j))

start = (0, 0)
goal = (4, 4)

# # Create maze and Monte Carlo RL agent with acceleration
# acceleration_maze = Race(width, height, start, goal, obstacles)
# mc_acceleration_rl_agent = MonteCarloAccelerationRL(acceleration_maze)
#
# # Train the agent
# num_episodes = 1000
# mc_acceleration_rl_agent.train(num_episodes)
#
# # Print the learned policy
# print("Learned Policy:")
# mc_acceleration_rl_agent.print_policy()

# plot the map
import matplotlib.pyplot as plt
plt.imshow(grid, cmap='hot', interpolation='nearest')
plt.show()



