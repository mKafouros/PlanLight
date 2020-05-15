import gym
import numpy as np
import cityflow

class TSCEnv(gym.Env):
    """
    Environment for Traffic Signal Control task.

    Parameters
    ----------
    world: World object
    agents: list of agent, corresponding to each intersection in world.intersections
    metric: Metric object, used to calculate evaluation metric
    """
    def __init__(self, world, agents, metric):
        self.world = world
        
        self.eng = self.world.eng
        self.n_agents = len(self.world.intersection_ids)
        self.n = self.n_agents

        # assert len(agents) == self.n_agents # comment it when parameter sharing

        self.agents = agents
        if len(self.agents) == self.n_agents:
            action_dims = [agent.action_space.n for agent in agents]
            self.action_space = gym.spaces.MultiDiscrete(action_dims)
        else:
            single_action = [len(world.intersections[0].phases)]
            action_dim = single_action*self.n_agents
            self.action_space = gym.spaces.MultiDiscrete(action_dim)
        self.metric = metric

    def step(self, actions):
        # print(actions)
        if not len(actions) == self.n_agents:
            print(actions)
        assert len(actions) == self.n_agents

        self.world.step(actions)

        if len(self.agents) == self.n_agents:
            obs = [agent.get_ob() for agent in self.agents]
            rewards = [agent.get_reward() for agent in self.agents]
        else:
            obs = self.agents[0].get_obs()
            rewards = self.agents[0].get_rewards()
        dones = [False] * self.n_agents
        #infos = {"metric": self.metric.update()}
        infos = {}

        return obs, rewards, dones, infos

    def reset(self):
        self.world.reset()
        if len(self.agents) == self.n_agents:
            obs = [agent.get_ob() for agent in self.agents]
        else:
            obs = self.agents[0].get_obs()
        return obs

    def get_current_obs(self):
        if len(self.agents) == self.n_agents:
            obs = [agent.get_ob() for agent in self.agents]
        else:
            obs = self.agents[0].get_obs()
        return obs

    def load_snapshot(self, archive=None, from_file=False, verbose=False, dir="./", file_name="snapshot"):
        if archive is not None:
            archive, intersection_infos = archive
        else:
            archive, intersection_infos = None, None

        return self.world.load_snapshot(archive, from_file=from_file, intersection_infos=intersection_infos,
                                        verbose=verbose, dir=dir, file_name=file_name)

    def take_snapshot(self, to_file=False, dir="./", file_name="snapshot", verbose=False):
        archive, intersection_infos = self.world.take_snapshot(to_file=to_file, dir=dir, verbose=verbose,
                                                               file_name=file_name)
        return [archive, intersection_infos]

    def reset_eng(self, config, thread_num):
        self.world.reset_eng(config, thread_num)
        self.eng = self.world.eng