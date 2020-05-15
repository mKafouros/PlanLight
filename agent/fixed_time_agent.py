from . import BaseAgent

class FixedTimeAgent(BaseAgent):
    """
    Agent using Max-Pressure method to control traffic light
    """
    def __init__(self, action_space, I, world, ob_generator=None, reward_generator=None):
        super().__init__(action_space)
        self.I = I
        self.world = world
        self.world.subscribe("lane_count")
        self.ob_generator = ob_generator
        self.reward_generator = reward_generator
        
        # the minimum duration of time of one phase
        self.t_min = 20
        self.last_action = -1
        self.len_actions = self.action_space.n

    def get_ob(self):
        if self.ob_generator is not None:
            return self.ob_generator.generate() 
        else:
            return None

    def get_action(self, ob):
        action = self.last_action + 1
        action %= self.len_actions
        self.last_action = action

        return action

    def get_reward(self):
        if self.reward_generator is None:
            return None
        else:
            reward = self.reward_generator.generate()
            assert len(reward) == 1
            return reward[0]