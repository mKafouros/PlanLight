import gym
from environment import TSCEnv
from world import World
from generator import LaneVehicleGenerator
from agent.dqn_agent import DQNAgent
from metric import TravelTimeMetric
import argparse
import os
import numpy as np
import logging
from datetime import datetime
from mute_tf_warnings import tf_mute_warning
from utils.util import generate_test_config
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, RMSprop, SGD

# parse args
parser = argparse.ArgumentParser(description='Run Example')
parser.add_argument('config_file', type=str, help='path of config file')
parser.add_argument('--thread', type=int, default=1, help='number of threads')
parser.add_argument('--steps', type=int, default=3600, help='number of steps')
parser.add_argument('--action_interval', type=int, default=20, help='how often agent make decisions')
parser.add_argument('--episodes', type=int, default=200, help='training episodes')
parser.add_argument('--share_weights', action="store_true", default=False)
parser.add_argument('--save_model', action="store_true", default=False)
parser.add_argument('--load_model', action="store_true", default=False)
parser.add_argument("--save_rate", type=int, default=10, help="save model once every time this many episodes are completed")
parser.add_argument('--save_dir', type=str, default="model-exp/single/dqn/", help='directory in which model should be saved')
parser.add_argument('--log_dir', type=str, default="log/dqn", help='directory in which logs should be saved')
args = parser.parse_args()
save_dir = args.save_dir
def init(args, test=False):
    tf_mute_warning()
    args.save_dir = save_dir + args.config_file[7:-5]
    if test:
        args.save_dir = save_dir + args.config_file[7:-10]

    # config_name = args.config_file.split('/')[1].split('.')[0]
    # args.agent_save_dir = args.save_dir + "/" + config_name
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(args.log_dir, datetime.now().strftime('%Y%m%d-%H%M%S') + ".log"))
    fh.setLevel(logging.DEBUG)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)

    # create world
    world = World(args.config_file, thread_num=args.thread, silent=True)

    # create agents
    agents = []
    for i in world.intersections:
        action_space = gym.spaces.Discrete(len(i.phases))
        agents.append(DQNAgent(
            action_space,
            LaneVehicleGenerator(world, i, ["lane_count"], in_only=True, average=None),
            LaneVehicleGenerator(world, i, ["lane_waiting_count"], in_only=True, average="all", negative=True),
            i.id
        ))
        if args.load_model:
            agents[-1].load_model(args.save_dir)
    if args.share_weights:
        model = agents[0].model
        for agent in agents:
            agent.model = model

    # create metric
    metric = TravelTimeMetric(world)

    # create env
    env = TSCEnv(world, agents, metric)

    return env

# train dqn_agent
def train(env, args):
    total_decision_num = 0
    best_result = evaluate(env)
    for e in range(args.episodes):
        last_obs = env.reset()

        episodes_rewards = [0 for i in env.agents]
        episodes_decision_num = 0
        i = 0
        while i < args.steps:
            if i % args.action_interval == 0:
                actions = []
                for agent_id, agent in enumerate(env.agents):
                    if args.share_weights:
                        agent = env.agents[0]
                    if total_decision_num > agent.learning_start:
                    #if True:
                        actions.append(agent.get_action(last_obs[agent_id]))
                    else:
                        actions.append(agent.sample())

                rewards_list = []
                for _ in range(args.action_interval):
                    obs, rewards, dones, _ = env.step(actions)
                    i += 1
                    rewards_list.append(rewards)
                rewards = np.mean(rewards_list, axis=0)

                for agent_id, agent in enumerate(env.agents):
                    if args.share_weights:
                        agent = env.agents[0]
                    agent.remember(last_obs[agent_id], actions[agent_id], rewards[agent_id], obs[agent_id])
                    episodes_rewards[agent_id] += rewards[agent_id]
                    episodes_decision_num += 1
                    total_decision_num += 1
                
                last_obs = obs

            for agent_id, agent in enumerate(env.agents):
                if args.share_weights:
                    if agent_id > 0:
                        break
                if total_decision_num > agent.learning_start and total_decision_num % agent.update_model_freq == agent.update_model_freq - 1:
                    agent.replay()
                if total_decision_num > agent.learning_start and total_decision_num % agent.update_target_model_freq == agent.update_target_model_freq - 1:
                    agent.update_target_network()
            if all(dones):
                break
        # if e % args.save_rate == args.save_rate - 1:
        #     if not os.path.exists(args.save_dir):
        #         os.makedirs(args.save_dir)
        #     for agent in agents:
        #         agent.save_model(args.save_dir)
        #         print("model saved")
        # if e % args.save_rate == 0:
        #     if not os.path.exists(args.save_dir):
        #         os.makedirs(args.save_dir)
        #     for agent in agents:
        #         agent.save_model(args.save_dir)
        #         print("model saved")
        current_result = evaluate(env)
        # print("current_result, episode:{}/{}, result:{}".format(e, args.episodes, current_result))
        if e % args.save_rate == 0:
            if current_result < best_result:
                for agent in env.agents:
                    if args.share_weights:
                        env.agents[0].save_model(args.save_dir)
                        break
                    agent.save_model(args.save_dir)
                best_result = current_result
                print("best model saved, episode:{}/{}, result:{}".format(e, args.episodes, current_result))
        # print("episode:{}/{}, average travel time:{}".format(e, args.episodes, env.eng.get_average_travel_time()))
        # for agent_id, agent in enumerate(agents):
        #     logger.info("agent:{}, mean_episode_reward:{}".format(agent_id, episodes_rewards[agent_id] / episodes_decision_num))

def evaluate(env):
    obs_n = env.reset()
    step = 0
    while step < args.steps:
        if step % args.action_interval == 0:
            # get action
            action_n = [agent.get_action(obs, test=True) for agent, obs in zip(env.agents, obs_n)] if not args.share_weights else [env.agents[0].get_action(obs, test=True) for obs in obs_n]
            for _ in range(args.action_interval):
                obs_n, rew_n, done_n, info_n = env.step(action_n)
                step += 1
    return env.eng.get_average_travel_time()


def test(env, args):
    obs = env.reset()
    for agent in env.agents:
        agent.load_model(args.save_dir)
    i = 0
    while i < args.steps:
        if i % args.action_interval == 0:
            actions = []
            for agent_id, agent in enumerate(env.agents):
                if args.share_weights:
                    agent = env.agents[0]
                actions.append(agent.get_action(obs[agent_id], test=True))
            for _ in range(args.action_interval):
                obs, rewards, dones, info = env.step(actions)
                i += 1
        #print(rewards)

        if all(dones):
            break
    return env.eng.get_average_travel_time()

if __name__ == '__main__':
    if args.config_file == "train-single":
        configs = ["config11syn1600", "config11syn2400", "config1-1", "config2-1", "config3-1", "config4-1",
                   "config5-1"]
        for config in configs:
            print(config)
            args.config_file = "config/" + config + ".json"
            env = init(args)
            train(env, args)
    elif args.config_file == "test":
        configs = ["config11syn1600", "config11syn2400", "config1-1", "config2-1", "config3-1", "config4-1",
                   "config5-1"]
        for config in configs:
            result_sum = 0.
            print(config)
            for i in range(10):
                args.config_file = "config/" + config + ".json"
                args.config_file = generate_test_config(args.config_file)
                env = init(args, test=True)
                result_sum += test(env, args)
            print("Average Travel Time is %.4f" % (result_sum/10))
    elif args.config_file == "train":
        args.episodes = 500
        # configs = ["config22syn", "config34-real", "config44-real3"]
        configs = ["config33mysyn", "config66mysyn"]
        for config in configs:
            print(config)
            args.config_file = "config/" + config + ".json"
            env = init(args)
            train(env, args)

