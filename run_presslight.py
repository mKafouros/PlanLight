import gym
from environment import TSCEnv
from world import World
from generator import LaneVehicleGenerator
from agent.presslight_agent import PressLightAgent
from metric import TravelTimeMetric
import argparse
import os
import numpy as np
import logging
from datetime import datetime
from utils.util import generate_test_config
from mute_tf_warnings import tf_mute_warning

# parse args
parser = argparse.ArgumentParser(description='Run Example')
parser.add_argument('config_file', type=str, help='path of config file')
parser.add_argument('--thread', type=int, default=4, help='number of threads')
parser.add_argument('--steps', type=int, default=3600, help='number of steps')
parser.add_argument('--action_interval', type=int, default=20, help='how often agent make decisions')
parser.add_argument('--episodes', type=int, default=200, help='training episodes')
parser.add_argument('--save_model', action="store_true", default=False)
parser.add_argument('--load_model', action="store_true", default=False)
parser.add_argument("--save_rate", type=int, default=10,
                    help="save model once every time this many episodes are completed")
parser.add_argument('--save_dir', type=str, default="model-exp/multi/press/", help='directory in which model should be saved')
parser.add_argument('--log_dir', type=str, default="log/presslight", help='directory in which logs should be saved')

args = parser.parse_args()
save_dir = args.save_dir

def init(args, test=False):
    tf_mute_warning()
    args.save_dir = save_dir + args.config_file[7:-5]
    if test:
        args.save_dir = save_dir + args.config_file[7:-10]

    # config_name = args.config_file.split('/')[1].split('.')[0]
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
        agents.append(PressLightAgent(
            action_space,
            LaneVehicleGenerator(world, i, ["lane_count"], in_only=True, average=None),
            LaneVehicleGenerator(world, i, ["lane_waiting_count"], in_only=True, average="all", negative=True),
            i.id,
            world
        ))
        if args.load_model:
            agents[-1].load_model(args.save_dir)
    # print(agents[0].ob_length)
    # print(agents[0].action_space)

    # create metric
    metric = TravelTimeMetric(world)

    # create env
    env = TSCEnv(world, agents, metric)

    return env


# train presslight_agent
def train(env, args):
    total_decision_num = 0
    best_result = evaluate(env)
    for e in range(args.episodes):
        last_obs = env.reset()
        # if e % args.save_rate == args.save_rate - 1:
        #     env.eng.set_save_replay(True)
        #     env.eng.set_replay_file("replay_%s.txt" % e)
        # else:
        #     env.eng.set_save_replay(False)
        episodes_rewards = [0 for i in env.agents]
        episodes_decision_num = 0
        i = 0
        while i < args.steps:
            if i % args.action_interval == 0:
                actions = []
                last_phase = []
                for agent_id, agent in enumerate(env.agents):
                    last_phase.append([env.world.intersections[agent_id].current_phase])
                    if total_decision_num > agent.learning_start:
                        # if True:
                        actions.append(
                            agent.get_action([env.world.intersections[agent_id].current_phase], last_obs[agent_id]))
                    else:
                        actions.append(agent.sample())

                rewards_list = []
                for _ in range(args.action_interval):
                    obs, rewards, dones, _ = env.step(actions)
                    i += 1
                    rewards_list.append(rewards)
                rewards = np.mean(rewards_list, axis=0)

                for agent_id, agent in enumerate(env.agents):
                    agent.remember(last_obs[agent_id], last_phase[agent_id], actions[agent_id], rewards[agent_id],
                                   obs[agent_id],
                                   [env.world.intersections[agent_id].current_phase])
                    episodes_rewards[agent_id] += rewards[agent_id]
                    episodes_decision_num += 1
                    total_decision_num += 1

                last_obs = obs

            for agent_id, agent in enumerate(env.agents):
                if total_decision_num > agent.learning_start and total_decision_num % agent.update_model_freq == agent.update_model_freq - 1:
                    agent.replay()
                if total_decision_num > agent.learning_start and total_decision_num % agent.update_target_model_freq == agent.update_target_model_freq - 1:
                    agent.update_target_network()
            if all(dones):
                break
        current_result = evaluate(env)
        print("current_result, episode:{}/{}, result:{}, avg_reward: {}".format(e, args.episodes, current_result, sum(episodes_rewards)/episodes_decision_num))
        if e % args.save_rate == 0:
            if current_result < best_result:
                for agent in env.agents:
                    agent.save_model(args.save_dir)
                best_result = current_result
                print("best model saved, episode:{}/{}, result:{}".format(e, args.episodes, current_result))
        # if e % args.save_rate == args.save_rate - 1:
        #     if not os.path.exists(args.save_dir):
        #         os.makedirs(args.save_dir)
        #     for agent in agents:
        #         agent.save_model(args.save_dir)
        # logger.info("episode:{}/{}, average travel time:{}".format(e, args.episodes, env.eng.get_average_travel_time()))
        # for agent_id, agent in enumerate(agents):
        #     logger.info(
        #         "agent:{}, mean_episode_reward:{}".format(agent_id, episodes_rewards[agent_id] / episodes_decision_num))


def evaluate(env):
    last_obs = env.reset()
    step = 0
    while step < args.steps:
        if step % args.action_interval == 0:
            # get action
            actions = []
            last_phase = []
            for agent_id, agent in enumerate(env.agents):
                last_phase.append([env.world.intersections[agent_id].current_phase])
                actions.append(
                    agent.get_action([env.world.intersections[agent_id].current_phase], last_obs[agent_id], test=True))
            for _ in range(args.action_interval):
                obs, rewards, dones, _ = env.step(actions)
                step += 1
            last_obs = obs
    return env.eng.get_average_travel_time()

def test(env, args):
    last_obs = env.reset()
    for agent in env.agents:
        agent.load_model(args.save_dir)
    for i in range(args.steps):
        if i % args.action_interval == 0:
            actions = []
            last_phase = []
            for agent_id, agent in enumerate(env.agents):
                last_phase.append([env.world.intersections[agent_id].current_phase])
                actions.append(
                    agent.get_action([env.world.intersections[agent_id].current_phase], last_obs[agent_id], test=True))
            rewards_list = []
            for _ in range(args.action_interval):
                obs, rewards, dones, _ = env.step(actions)
                i += 1
                rewards_list.append(rewards)
            last_obs = obs
            rewards = np.mean(rewards_list, axis=0)
        # print(env.eng.get_average_travel_time())
        if all(dones):
            break
    return env.eng.get_average_travel_time()


def meta_test(config):
    obs = env.reset()
    last_obs = obs
    # env.change_world(World(config, thread_num=args.thread))
    for agent in env.agents:
        agent.load_model(args.save_dir)
    total_decision_num = 0
    for i in range(args.steps):
        if i % args.action_interval == 0:
            actions = []
            last_phase = []
            for agent_id, agent in enumerate(env.agents):
                last_phase.append([env.world.intersections[agent_id].current_phase])
                actions.append(agent.get_action([env.world.intersections[agent_id].current_phase], obs[agent_id]))
            rewards_list = []
            for _ in range(args.action_interval):
                obs, rewards, dones, _ = env.step(actions)
                i += 1
                rewards_list.append(rewards)
            rewards = np.mean(rewards_list, axis=0)
            for agent_id, agent in enumerate(agents):
                agent.remember(last_obs[agent_id], last_phase[agent_id], actions[agent_id], rewards[agent_id],
                               obs[agent_id],
                               [env.world.intersections[agent_id].current_phase])
                total_decision_num += 1
            last_obs = obs
        for agent_id, agent in enumerate(agents):
            if total_decision_num % agent.update_model_freq == agent.update_model_freq - 1:
                agent.replay()
            if total_decision_num % agent.update_target_model_freq == agent.update_target_model_freq - 1:
                agent.update_target_network()
        # print(env.eng.get_average_travel_time())
        if all(dones):
            break
    logger.info("Final Travel Time is %.4f" % env.eng.get_average_travel_time())

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
    elif args.config_file == "exp":
        configs = ["config22syn", "config33mysyn", "config66mysyn", "config44-real3", "config34-real"]
        save_dir = "./model-exp/single/press"
        for config in configs:
            args.config_file = "config/" + config + ".json"
            args.save_dir = save_dir + args.config_file[7:-5]
            print(config)
            env = init(args)
            result = test(env, args)
            print("Average Travel Time is %.4f" % result)
    else:
        env = init(args)
        train(env, args)