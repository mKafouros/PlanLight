import sys
sys.path.append("../")

import gym
from agent import BCCoLightAgent, MaxPressureAgent, BehaviorCloningAgent, GailAgent, CoLightAgent, FRAP_DQNAgent
from world import World
from environment import TSCEnv
from generator import LaneVehicleGenerator, PressurePhaseGenerator
from metric.travel_time import TravelTimeMetric

import random
from .trajectories_buffer import TrajectoryBuffer
from .util import argmin, build_int_intersection_map
import multiprocessing
import time
from mute_tf_warnings import tf_mute_warning
import os
import cityflow
import json
# import tensorflow as tf
import logging

class RolloutControllerSingle(object):

    def __init__(self, env, args):
        self.env = env
        self.args = args

    def simulate_one_action(self, actions, current_step):
        i = current_step
        for _ in range(self.args.action_interval):
            obs, rewards, dones, info = self.env.step(actions)
            i += 1
        while i < self.args.steps:
            actions = []
            for agent_id, agent in enumerate(self.env.agents):
                actions.append(agent.get_action(obs[agent_id]))
            for _ in range(self.args.action_interval):
                obs, rewards, dones, info = self.env.step(actions)
                i += 1
        # print("rollout over, current step: {0}, time: {1}".format(i, world.eng.get_current_time()))
        result = self.env.eng.get_average_travel_time()
        # result = info["metric"]
        return result

    def rollout_one_step(self, archive, current_step, verbose=True):
        t0 = time.time()
        results = []
        for action in range(self.env.agents[0].action_space.n):
            # for action in range(1):
            self.env.load_snapshot(archive, verbose=False)
            actions = [action]
            result = self.simulate_one_action(actions, current_step)
            results.append(result)
        # print(results)
        best_action = argmin(results)
        best_value = results[best_action]
        t1 = time.time()
        # print(best_action)
        if verbose or current_step > 3500:
            print("best value: {}, time: {}".format(results[best_action], t1 - t0))
        self.env.load_snapshot(archive)
        return best_action

    def perform_rollout(self, trj_buffer=None, random_walk=False, walk_rate=0.2):
        action_history = []
        obs = self.env.reset()
        i = 0
        if trj_buffer is None:
            trj_buffer = TrajectoryBuffer(10000, file_name="latest_unknown_trajectory")
        while i < self.args.steps:
            if i%600 == 0:
                print("current step: %d" % i)

            # print(trj_buffer.__len__())
            archive = self.env.take_snapshot(verbose=False)
            actions = [self.rollout_one_step(archive, i, verbose=(i%600==0))]
            if random_walk and walk_rate > random.random():
                actions[0] = self.env.agents[0].sample()
            action_history.append(actions[0])


            # print("actual action for step {0}: {1}".format(i, actions[0]))
            trj_buffer.add(obs[0], actions[0])
            for _ in range(self.args.action_interval):
                obs, rewards, dones, info = self.env.step(actions)
                i += 1
            # print(action_history)

def create_agents(args, world, policy):
    agents = []
    if policy == "max_pressure":
        for i in world.intersections:
            action_space = gym.spaces.Discrete(len(i.phases))
            agents.append(MaxPressureAgent(
                action_space, i, world
            ))
    elif policy == "bc":
        for i in world.intersections:
            action_space = gym.spaces.Discrete(len(i.phases))
            agents.append(BehaviorCloningAgent(
                action_space,
                PressurePhaseGenerator(world, i),
                i.id,
                args,
            ))
    elif policy == "gail":
        for i in world.intersections:
            action_space = gym.spaces.Discrete(len(i.phases))
            agents.append(GailAgent(
                action_space,
                LaneVehicleGenerator(world, i, ["lane_count"], in_only=True, average="lane"),
                i.id,
                args
            ))
    elif policy == "frap":
        for i in world.intersections:
            action_space = gym.spaces.Discrete(len(i.phases))
            agents.append(FRAP_DQNAgent(
                action_space,
                LaneVehicleGenerator(world, i, ["lane_count"], in_only=True, average=None),
                LaneVehicleGenerator(world, i, ["lane_waiting_count"], in_only=True, average="all", negative=True),
                world,
                i.id
            ))
    else:
        raise NotImplementedError
    return agents

def create_env(args, policy="max_pressure", agents=None):
    if policy=="bc_colight" or policy=="colight":
        env = create_colight_env(args, agent=policy)
        return  env
    world = World(args.config_file, thread_num=args.thread, silent=True)

    # create agents
    if agents is None:
        agents = create_agents(args, world, policy)

    # create metric
    metric = TravelTimeMetric(world)

    # create env
    env = TSCEnv(world, agents, metric)

    return env

def create_colight_env(args, agent="bc_colight"):
    config = json.load(open(args.config_file, "r"))
    road_net_file_path = config["dir"] + config["roadnetFile"]
    res = build_int_intersection_map(road_net_file_path)
    net_node_dict_id2inter = res[0]
    net_node_dict_inter2id = res[1]
    net_edge_dict_id2edge = res[2]
    net_edge_dict_edge2id = res[3]
    node_degree_node = res[4]
    node_degree_edge = res[5]
    node_adjacent_node_matrix = res[6]
    node_adjacent_edge_matrix = res[7]
    edge_adjacent_node_matrix = res[8]

    # create world
    world = World(args.config_file, thread_num=args.thread, silent=True)

    dic_traffic_env_conf = {
        "NUM_INTERSECTIONS": len(net_node_dict_id2inter),  # used
        "NUM_ROADS": len(net_edge_dict_id2edge),  # used
    }

    dic_graph_setting = {
        "NEIGHBOR_NUM": 4,  # standard number of adjacent nodes of each node
        "NEIGHBOR_EDGE_NUM": 4,  # # standard number of adjacent edges of each node
        "N_LAYERS": 1,  # layers of MPNN
        "INPUT_DIM": [128, 128],
    # input dimension of each layer of multiheadattention, the first value should == the last value of "NODE_EMB_DIM"
        "OUTPUT_DIM": [128, 128],
    # output dimension of each layer of multiheadattention, the first value should == the last value of "NODE_EMB_DIM"
        "NODE_EMB_DIM": [128, 128],  # the firsr two layer of dense to embedding the input
        "NUM_HEADS": [5, 5],
        "NODE_LAYER_DIMS_EACH_HEAD": [16, 16],  # [input_dim,output_dim]
        "OUTPUT_LAYERS":[], #
        "NEIGHBOR_ID": node_adjacent_node_matrix,  # adjacent node id of each node
        "ID2INTER_MAPPING": net_node_dict_id2inter,  # id ---> intersection mapping
        "INTER2ID_MAPPING": net_node_dict_inter2id,  # intersection ----->id mapping
        "NODE_DEGREE_NODE": node_degree_node,  # number of adjacent nodes of node
    }
    tmp_agents = []
    observation_generators = []
    for node_dict in world.intersections:
        node_id = node_dict.id
        action_space = gym.spaces.Discrete(len(node_dict.phases))
        node_id_int = net_node_dict_inter2id[node_id]
        tmp_generator = LaneVehicleGenerator(world,
                                             node_dict, ["lane_count"],
                                             in_only=True,
                                             average=None)
        observation_generators.append((node_id_int, tmp_generator))
    sorted(observation_generators,
           key=lambda x: x[0])  # sorted the ob_generator based on its corresponding id_int, increasingly
    # create agent
    action_space = gym.spaces.Discrete(len(world.intersections[0].phases))
    if agent == "bc_colight":
        colightAgent = BCCoLightAgent(
            action_space,
            observation_generators,
            world,
            dic_traffic_env_conf,
            dic_graph_setting,
            args
        )
    elif agent == "colight":
        colightAgent = CoLightAgent(
            action_space,
            observation_generators,
            LaneVehicleGenerator(world, world.intersections[0], ["lane_waiting_count"], in_only=True, average="all", negative=True),
            world,
            dic_traffic_env_conf,
            dic_graph_setting,
            args
        )
    else:
        colightAgent = None
    # print(colightAgent.ob_length)
    # print(colightAgent.action_space)
    # create metric
    metric = TravelTimeMetric(world)
    agents = [colightAgent]
    # create env
    env = TSCEnv(world, agents, metric)
    return env

class RolloutControllerMulti(object):
    def __init__(self, env, args):
        self.env = env
        self.args = args

    def simulate_one_action(self, actions, current_step):
        i = current_step
        for _ in range(self.args.action_interval):
            obs, rewards, dones, info = self.env.step(actions)
            i += 1
        while i < self.args.steps:
            actions = []
            for agent_id, agent in enumerate(self.env.agents):
                actions.append(agent.get_action(obs[agent_id]))
            for _ in range(self.args.action_interval):
                obs, rewards, dones, info = self.env.step(actions)
                i += 1
        # print("rollout over, current step: {0}, time: {1}".format(i, world.eng.get_current_time()))
        result = self.env.eng.get_average_travel_time()
        # result = info["metric"]
        return result

    def rollout_one_step(self, archive, current_step, agent_order, verbose=True):
        t0 = time.time()
        default_actions = [-1 for i in range(len(self.env.agents))]
        rollout_result = {}
        obs = self.env.get_current_obs()
        for agent_id, agent in enumerate(self.env.agents):
            default_actions[agent_id] = agent.get_action(obs[agent_id])
            rollout_result[agent_id] = None

        for current_agent_id in agent_order:
            # print("step: {}, agent: {} is running...".format(current_step, current_agent_id))
            results = []
            for action in range(self.env.agents[current_agent_id].action_space.n):
                # print("agent: {}, action: {} is running".format(current_agent_id, action))
                self.env.load_snapshot(archive, verbose=False)
                actions = [default_actions[i] if rollout_result[i] is None else rollout_result[i] for i in range(len(self.env.agents))]
                actions[current_agent_id] = action
                result = self.simulate_one_action(actions, current_step)
                results.append(result)
            best_action = argmin(results)
            best_value = results[best_action]
            # if verbose:
            #     print("step: {}, agent: {}, best value: {}".format(current_step, current_agent_id, best_value))
            rollout_result[current_agent_id] = best_action
        t1 = time.time()
        if verbose:
            print("step: {}, best value: {}, time: {}".format(current_step, best_value, t1-t0))
        self.env.load_snapshot(archive)
        # print([rollout_result[i] for i in range(len(self.env.agents))])
        return [rollout_result[i] for i in range(len(self.env.agents))]

    def perform_rollout(self, trj_buffer_list=None, random_walk=False, walk_rate=0.2, save_trj=True, model_dirs=[], model_id=0):
        # action_history = []
        obs = self.env.reset()
        i = 0

        if trj_buffer_list is None:
            trj_buffer_list = [TrajectoryBuffer(10000, file_name="latest_unknown_trajectory") for i in range(len(self.env.agents))]
        elif len(self.env.agents) == 1 and isinstance(trj_buffer_list, TrajectoryBuffer):
            trj_buffer_list = [trj_buffer_list]

        while i < self.args.steps:
            archive = self.env.take_snapshot(verbose=False)

            agent_order = [i for i in range(len(self.env.agents))]
            random.shuffle(agent_order)
            actions = self.rollout_one_step(archive, i, agent_order, verbose=(i % 300 == 0))
            # actions = self.rollout_one_step(archive, i, agent_order, verbose=True)
            for agent_id, agent in enumerate(self.env.agents):
                if random_walk and walk_rate > random.random():
                    random_action = agent.sample()
                    actions[agent_id] = random_action
                if save_trj:
                    trj_buffer_list[agent_id].add(obs[agent_id], actions[agent_id])

            # print("actual action for step {0}: {1}".format(i, actions[0]))
            for _ in range(self.args.action_interval):
                obs, rewards, dones, info = self.env.step(actions)
                i += 1
            # print(action_history)

class RolloutProcess(multiprocessing.Process):
    def __init__(self, process_id, args, action_queue, result_queue, policy="max_pressure", archive_name="snapshot"):
        super(RolloutProcess, self).__init__()
        self.process_id = process_id
        self.args = args
        self.action_queue = action_queue
        self.result_queue = result_queue
        self.policy = policy
        self.archive_name = archive_name

    def run(self):
        def rollout_single_action(args, env):
            obs = env.get_current_obs()
            start_step = env.eng.get_current_time()
            i = env.eng.get_current_time()
            while i < args.steps:
                if args.time_horizon:
                    if i >= start_step + args.time_horizon:
                        break
                actions = []
                for agent_id, agent in enumerate(env.agents):
                    if self.args.parameter_sharing:
                        actions = env.agents[0].get_actions(obs)
                        break
                    actions.append(agent.get_action(obs[agent_id]))
                for _ in range(args.action_interval):
                    obs, rewards, dones, info = env.step(actions)
                    i += 1
            result = env.eng.get_average_travel_time()
            return result
        import time
        t0 = time.time()
        import tensorflow as tf
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.env = create_env(self.args, policy=self.policy)
        self.n_intersections = len(self.env.world.intersections)
        t1 = time.time()
        # print("environment for process {} created!, time: {}".format(self.process_id, t1 - t0))
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        tf_mute_warning()

        while True:
            if not self.action_queue.empty():
                actions = self.action_queue.get()
            else:
                time.sleep(1)
                continue

            if actions == "end":
                break
            elif isinstance(actions, dict):
                model_id = actions["model_id"]
                model_dirs = actions["model_dirs"]
                with self.graph.as_default():
                    if not self.policy == "max_pressure":
                        if self.policy == "frap":
                            for agent in self.env.agents:
                                agent.load_model(dir=model_dirs[0])
                        elif self.args.parameter_sharing:
                            self.env.agents[0].load_model(dir=model_dirs[0], model_id=model_id)
                        else:
                            for agent_id in range(len(model_dirs)):
                                self.env.agents[agent_id].load_model(dir=model_dirs[agent_id], model_id=model_id)
                time.sleep(1)
            else:
                self.env.load_snapshot(from_file=True, dir="./archive", file_name=self.archive_name)
                obs = self.env.get_current_obs()
                with self.graph.as_default():
                    if self.args.parameter_sharing:
                        default_actions = self.env.agents[0].get_actions(obs)
                        # print(default_actions)
                    # print(actions)
                    for agent_id in range(self.n_intersections):
                        if actions[agent_id] is None:
                            if self.args.parameter_sharing:
                                actions[agent_id] = default_actions[agent_id]
                            else:
                                actions[agent_id] = self.env.agents[agent_id].get_action(obs[agent_id])
                        elif isinstance(actions[agent_id], str):
                            actions[agent_id] = int(actions[agent_id])
                            rollout_action_id = actions[agent_id]
                for _ in range(self.args.action_interval):
                    # print(actions)
                    self.env.step(actions)
                with self.graph.as_default():
                    result = rollout_single_action(self.args, self.env)
                self.result_queue.put([rollout_action_id, result])

class RolloutControllerParallel(object):
    def __init__(self, args, base_policy="max_pressure", rollout_agent_num=0):
        self.args = args
        self.base_policy = base_policy
        self.tmp_archive_name = self.args.prefix+"_snapshot"
        tmp_world = World(args.config_file, thread_num=args.thread, silent=True)
        self.n_intersections = len(tmp_world.intersections)
        self.action_space = len(tmp_world.intersections[0].phases)
        self.rollout_agent_num = self.n_intersections if rollout_agent_num <= 0 else rollout_agent_num
        print("creating subprocesses....")
        self.action_queue = multiprocessing.Queue(self.action_space)
        self.result_queue = multiprocessing.Queue(self.action_space)
        tf_mute_warning()
        self.pool = []
        for process_id in range(self.action_space):
            process = RolloutProcess(process_id, self.args, self.action_queue, self.result_queue,
                                     policy=self.base_policy, archive_name=self.tmp_archive_name)
            self.pool.append(process)
        for p in self.pool:
            p.start()


        self.env = create_env(args, policy=base_policy)

        # agents = create_agents(args, env.world, base_policy)
        # env.agents = agents

    def get_env(self):
        return self.env

    def perform_rollout(self, trj_buffer_list=None, random_walk=False, walk_rate=0.2, save_trj=True, model_dirs=[], model_id=0, verbose=0, save_traj=False):
        # action_history = []
        obs = self.env.reset()

        if trj_buffer_list is None:
            trj_buffer_list = [TrajectoryBuffer(10000, file_name="latest_unknown_trajectory") for i in range(len(self.env.agents))]
        elif len(self.env.agents) == 1 and isinstance(trj_buffer_list, TrajectoryBuffer):
            trj_buffer_list = [trj_buffer_list]
        if not self.base_policy=="max_pressure":
            for i in range(self.action_space):
                model_dict = {}
                model_dict["model_id"] = model_id
                model_dict["model_dirs"] = model_dirs
                self.action_queue.put(model_dict)

        print("rollouting..")
        i = 0
        while i < self.args.steps:
            agent_order = [j for j in range(self.n_intersections)]
            random.shuffle(agent_order)
            agent_to_rollout = agent_order[:self.rollout_agent_num]
            if self.args.rollout_neighbor:
                agent_to_rollout = self.find_neighbors(random.randint(0, self.n_intersections - 1))
                # while len(agent_to_rollout) < 4:
                #     agent_to_rollout = self.find_neighbors(random.randint(0, self.n_intersections - 1))
            try:
                self.env.take_snapshot(to_file=True, dir="./archive", file_name=self.tmp_archive_name)
                # default_actions = []
                # for agent_id, agent in enumerate(self.env.agents):
                #     if self.args.parameter_sharing:
                #         default_actions = self.env.agents[0].get_actions(obs).tolist()
                #         break
                #     default_actions.append(agent.get_action(obs[agent_id]))
                actions = self.rollout_one_step(i, agent_order, agent_to_rollout, verbose=(i % verbose == 0) if verbose else False)
                # target = "old" if actions == default_actions else "new"
                target = None
                if self.args.parameter_sharing:
                    if self.base_policy == "bc_colight" or self.base_policy == "colight":
                        trj_buffer_list[0].add(obs, actions, save=save_traj, target=target)
                    else:
                        for agent_id in agent_to_rollout:
                            trj_buffer_list[0].add(obs[agent_id], actions[agent_id], save=save_traj)
                for agent_id in range(self.n_intersections):
                    if save_trj and not self.args.parameter_sharing:
                        trj_buffer_list[0].add(obs[agent_id], actions[agent_id], save=save_traj)
                    if random_walk and walk_rate > random.random():
                        if self.args.parameter_sharing:
                            random_action = self.env.agents[0].sample()
                        else:
                            random_action = self.env.agents[agent_id].sample()
                        actions[agent_id] = random_action
            except json.decoder.JSONDecodeError:
                actions = []
                print("skipping rollout step{} due to archive failure".format(i))
                if self.args.parameter_sharing:
                    actions = self.env.agents[0].get_actions(obs)
                else:
                    for agent_id, agent in enumerate(self.env.agents):
                        actions.append(agent.get_action(obs[agent_id]))

            # print("actual action for step {0}: {1}".format(i, actions[0]))
            for _ in range(self.args.action_interval):
                obs, rewards, dones, info = self.env.step(actions)
                i += 1

        return 1


    def rollout_one_step(self, current_step, agent_order, agent_to_rollout, verbose=True):
        t0 = time.time()
        rollout_result = {}
        for agent_id in range(self.n_intersections):
            rollout_result[agent_id] = None
        obs = self.env.get_current_obs()

        self.env.load_snapshot(from_file=True, dir="./archive", file_name=self.tmp_archive_name)
        for current_agent_id in agent_order:
            if current_agent_id not in agent_to_rollout:
                if self.args.parameter_sharing:
                    default_actions = self.env.agents[0].get_actions(obs)
                    rollout_result[current_agent_id] = default_actions[current_agent_id]
                else:
                    rollout_result[current_agent_id] = self.env.agents[current_agent_id].get_action(obs[current_agent_id])
                continue
            process_actions = []
            for action in range(self.action_space):
                actions = [rollout_result[i] for i in range(self.n_intersections)]
                actions[current_agent_id] = str(action)
                process_actions.append(actions)
            for i in range(self.action_space):
                self.action_queue.put(process_actions[i])

            # print("waiting process to rollout..")
            results = [0 for k in range(8)]
            count = 0
            while True:
                if count >= 8:
                    break
                if not self.result_queue.empty():
                    result = self.result_queue.get()
                    count += 1
                    # print(result)
                    results[result[0]] = result[1]
                else:
                    time.sleep(1)
            # print(rollout_result)
            best_action = argmin(results)
            best_value = results[best_action]
            rollout_result[current_agent_id] = best_action
            # if verbose:
            #     print("step: {}, agent: {}, results: {}".format(current_step, current_agent_id, results))
        t1 = time.time()
        if verbose or current_step > 3500:
            print("step: {}, best value: {}, time: {}".format(current_step, best_value, t1-t0))
            print("results: {}".format(results))
            logging.info("step: {}, best value: {}, time: {}".format(current_step, best_value, t1-t0))
        # self.env.load_snapshot(archive)
        # print([rollout_result[i] for i in range(self.n_intersections)])
        return [rollout_result[i] for i in range(self.n_intersections)]

    def find_neighbors(self, id):
        neighbors = [id]
        intersection_map = self.env.world.intersection_map
        dim = self.env.world.intersections_dim
        agent_pos = self.env.world.intersection_positions[id]
        agent_index = [agent_pos[0] - 1, agent_pos[1] - 1]
        if agent_index[0] > 0:
            neighbors.append(intersection_map[agent_index[0] - 1][agent_index[1]])
        if agent_index[0] < dim[0] - 1:
            neighbors.append(intersection_map[agent_index[0] + 1][agent_index[1]])
        if agent_index[1] > 0:
            neighbors.append(intersection_map[agent_index[0]][agent_index[1] - 1])
        if agent_index[1] < dim[1] - 1:
            neighbors.append(intersection_map[agent_index[0]][agent_index[1] + 1])
        return list(map(int, neighbors))

    def __del__(self):
        print("all rollout finished, waiting processes to quit")
        for i in range(self.action_space):
            self.action_queue.put("end")
        for p in self.pool:
            p.join()
