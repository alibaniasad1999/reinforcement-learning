import numpy as np
import matplotlib.pyplot as plt

# solving car racing problem using monte carlo method reinforcement learning
# the car racing problem is a 5x5 grid, with the car starting at the bottom
# left corner, and the goal at the top right corner. The car can move up, down, left, or right


# initialize the grid
grid = np.zeros((5,5))
# set the goal
grid[4,4] = 1
# set the starting position down in random row
# grid[4,np.random.randint(0,5)] = 2
# set walls
grid[1,1] = -1
grid[1,2] = -1
grid[2,1] = -1
grid[2,2] = -1
# grid[3,2] = -1
# grid[3,3] = -1
# grid[3,4] = -1

# plot the grid revers y axis
# plt.imshow(grid, cmap='gray')
# plt.gca().invert_yaxis()
# plt.show()

import numpy as np
import matplotlib.pyplot as plt


class Race:
    def __init__(self, grid):
        self.reward = None
        self.grid = grid

        self.state = np.array([np.random.randint(0, len(grid)), 0, 0, 0])  # x y dx dy
        self.returns = {(i, j, k, l, m, n): [] for i in range(len(grid)) for j in range(len(grid)) for k in range(-5, 6)
                        for l in range(6) for m in range(-1, 2) for n in range(-1, 2)}
        self.Q = {(i, j, k, l, m, n): 0 for i in range(len(grid)) for j in range(len(grid)) for k in range(-5, 6) for l
                  in range(6) for m in range(-1, 2) for n in range(-1, 2)}
        # random policy
        self.policy = {(i, j, k, l): np.random.randint(-1, 2, size=2) for i in range(len(grid)) for j in
                       range(len(grid)) for k in range(-5, 6) for l in range(6)}

        self.epsilon = 0.1
        self.alpha = 0.1
        self.gamma = 0.9
        self.is_done = False

    def get_action(self):
        # greedy policy
        if np.random.uniform(0, 1) > self.epsilon:
            return self.policy[self.state[0], self.state[1], self.state[2], self.state[3]]
        else:
            return np.random.randint(-1, 2, size=2)

    def reset(self):
        self.state = np.array([np.random.randint(0, len(grid)), 0, 0, 0])
        self.is_done = False

    def step(self, action):
        # move the car
        self.state[0] += self.state[2]  # x + dx
        self.state[1] += self.state[3]  # y + dy
        self.state[2] += action[0]
        self.state[3] += action[1]
        # check velocity
        if self.state[2] > 5:
            self.state[2] = 5
        elif self.state[2] < -5:
            self.state[2] = -5
        if self.state[3] > 5:
            self.state[3] = 5
        elif self.state[3] < 0:
            self.state[3] = 1
        # check if the car is out of bounds
        if self.state[0] < 0:
            self.state[0] = 0
            self.state[2] = 0
        elif self.state[0] > len(self.grid) - 1:
            self.state[0] = len(self.grid) - 1
            self.state[2] = 0
        if self.state[1] < 0:
            self.state[1] = 0
            self.state[3] = 0
        elif self.state[1] > len(self.grid) - 1:
            self.state[1] = len(self.grid) - 1
            self.state[3] = 0
        # check if the car hit a wall
        if self.grid[self.state[0], self.state[1]] == -1:
            self.is_done = True
            self.reward = -5
        # check if the car hit the goal
        elif self.grid[self.state[0], self.state[1]] == 1:
            self.is_done = True
            self.reward = 5
        else:
            self.is_done = False
            self.reward = -1

        # plot grid with car
        plt.imshow(self.grid, cmap='gray')
        plt.gca().invert_yaxis()
        plt.scatter(self.state[0], self.state[1], color='red')
        plt.show()

        print('State: ', self.state)
        return self.state, self.reward, self.is_done

    def play(self, episodes):
        for episode in range(episodes):
            # reset the car
            self.reset()
            # play the game
            while not self.is_done:
                # get the action
                action = self.get_action()
                # step the car
                state, reward, is_done = self.step(action)
                # add the reward to the returns
                self.returns[(state[0], state[1], state[2], state[3], action[0], action[1])].append(reward)
            # update the Q values
            for key in self.returns:
                # check is empty
                if len(self.returns[key]) > 0:
                    # update the Q value
                    self.Q[key] = np.mean(self.returns[key])
            # update the policy to be greedy to max Q
            for key in self.policy:
                for i in range(-1, 2):
                    max = -np.inf
                    for j in range(-1, 2):
                        # find max Q to update policy
                        if self.Q[(key[0], key[1], key[2], key[3], i, j)] > max:
                            max = self.Q[(key[0], key[1], key[2], key[3], i, j)]
                            self.policy[(key[0], key[1], key[2], key[3])] = np.array([i, j])
            # print the policy
            print('Episode: ', episode)
            # print('Policy: ', self.policy)
            # print('Q: ', self.Q)
            # print('Returns: ', self.returns)
            # print('----------------------------------------')
            # reset the returns
            self.returns = {(i, j, k, l, m, n): [] for i in range(len(grid)) for j in range(len(grid)) for k in
                            range(-5, 6) for l in range(6) for m in range(-1, 2) for n in range(-1, 2)}

# play the game
game = Race(grid)
game.play(10)