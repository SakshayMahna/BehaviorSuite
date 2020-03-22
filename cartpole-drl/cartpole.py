import gym
import numpy as np
from dqn import DQN 

class Cartpole():
    def __init__(self):
        self.env = gym.make('CartPole-v1')
        self.observation_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.n

    def solve(self):
        solver = DQN(self.observation_space, self.action_space)
        run = 0

        while True:
            run = run + 1
            state = self.env.reset()
            state = np.reshape(state, [1, self.observation_space])

            step = 0
            while True:
                step = step + 1
                action = solver.act(state)
                state_next, reward, terminal, info = self.env.step(action)
                reward = reward if not terminal else -1 * reward
                state_next = np.reshape(state_next, [1, self.observation_space])
                solver.remember(state, action, reward, state_next, terminal)
                state = state_next

                if terminal:
                    print("Run: " + str(run) + "  Exploration: " + str(solver.exploration_rate) + " Score: " + str(step))
                    break

                solver.experience_replay()

if __name__ == '__main__':
    c = Cartpole()
    c.solve()
