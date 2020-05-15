print("loading modules....")
import gym
from environment import TSCEnv
from world import World
from generator import LaneVehicleGenerator
from agent.max_pressure_agent import MaxPressureAgent
from agent.bc_colight_agent import BCCoLightAgent
from agent.colight_agent import CoLightAgent
from mute_tf_warnings import tf_mute_warning
from metric import TravelTimeMetric
from utils.arguments import parse_args
from utils.util import build_int_intersection_map
import json
from utils.gail_trainer import train_bc_multi, test_multi, train_bc_share_weights
from utils.util import generate_test_config



#
# args = parse_args()
#
# config = json.load(open(args.config_file,"r"))
# road_net_file_path = config["dir"] + config["roadnetFile"]
# res = build_int_intersection_map(road_net_file_path)
# net_node_dict_id2inter = res[0]
# net_node_dict_inter2id =res[1]
# net_edge_dict_id2edge=res[2]
# net_edge_dict_edge2id=res[3]
# node_degree_node=res[4]
# node_degree_edge=res[5]
# node_adjacent_node_matrix=res[6]
# node_adjacent_edge_matrix=res[7]
# edge_adjacent_node_matrix=res[8]
#
# traj_buffer_list = [TrajectoryBuffer(10000, file_name="4X4_headstart_max_pressure_bc_colight")]
# # create world
# world = World(args.config_file, thread_num=args.thread)
#
# dic_traffic_env_conf = {
#     "NUM_INTERSECTIONS":len(net_node_dict_id2inter),  #used
#     "NUM_ROADS":len(net_edge_dict_id2edge),  #used
# }
#
# dic_graph_setting = {
#     "NEIGHBOR_NUM": 4,  # standard number of adjacent nodes of each node
#     "NEIGHBOR_EDGE_NUM": 4,  # # standard number of adjacent edges of each node
#     "N_LAYERS": 1,  # layers of MPNN
#     "INPUT_DIM": [128,128], # input dimension of each layer of multiheadattention, the first value should == the last value of "NODE_EMB_DIM"
#     "OUTPUT_DIM": [128,128], # output dimension of each layer of multiheadattention, the first value should == the last value of "NODE_EMB_DIM"
#     "NODE_EMB_DIM": [128,128],  # the firsr two layer of dense to embedding the input
#     "NUM_HEADS": [5,5],
#     "NODE_LAYER_DIMS_EACH_HEAD":[16, 16],  # [input_dim,output_dim]
#     "NEIGHBOR_ID": node_adjacent_node_matrix,  # adjacent node id of each node
#     "ID2INTER_MAPPING": net_node_dict_id2inter,  # id ---> intersection mapping
#     "INTER2ID_MAPPING":net_node_dict_inter2id,  # intersection ----->id mapping
#     "NODE_DEGREE_NODE": node_degree_node,  # number of adjacent nodes of node
# }
# tmp_agents = []
# observation_generators = []
# for node_dict in world.intersections:
#     node_id = node_dict.id
#     action_space = gym.spaces.Discrete(len(node_dict.phases))
#     node_id_int = net_node_dict_inter2id[node_id]
#     tmp_agent = MaxPressureAgent(
#         action_space, node_dict, world,
#         LaneVehicleGenerator(world, node_dict, ["lane_count"], in_only=True)
#     )
#     tmp_generator = LaneVehicleGenerator(world,
#                                          node_dict, ["lane_count"],
#                                          in_only=True,
#                                          average=None)
#     observation_generators.append((node_id_int, tmp_generator))
#     tmp_agents.append((node_id_int, tmp_agent))
# sorted(observation_generators, key=lambda x: x[0]) # sorted the ob_generator based on its corresponding id_int, increasingly
# sorted(tmp_agents, key=lambda x: x[0]) # sorted the ob_generator based on its corresponding id_int, increasingly
# expert_agents = []
# for agent in tmp_agents:
#     expert_agents.append(agent[1])
# # create agent
# action_space = gym.spaces.Discrete(len(world.intersections[0].phases))
# colightAgent = BCCoLightAgent(
#     action_space,
#     observation_generators,
#     world,
#     dic_traffic_env_conf,
#     dic_graph_setting,
#     args
# )
# # if args.load_model:
# #     colightAgent.load_model(args.load_dir)
# print(colightAgent.ob_length)
# print(colightAgent.action_space)
# # create metric
# metric = TravelTimeMetric(world)
# agents = [colightAgent]
# # create env
# env = TSCEnv(world, agents, metric)
# # simulate
# # print(len(world.intersections))
# obs = env.reset()
# actions = []
# i = 0
# while i < args.steps:
#     if i % args.action_interval == 0:
#         actions = []
#         for agent_id, agent in enumerate(expert_agents):
#             actions.append(agent.get_action(obs[agent_id]))
#         traj_buffer_list[0].add(obs, actions)
#         for _ in range(args.action_interval):
#             obs, rewards, dones, info = env.step(actions)
#             i += 1
#         print(obs.shape)
#         # print(obs.shape)
# print(len(traj_buffer_list[0]))
# print(env.eng.get_average_travel_time())
# traj_buffer_list[0].save_to_file()

# for i in range(args.episodes):
#     train_bc_share_weights(args, env, traj_buffer_list, ["./model/colight"], iteration_id=i, get_best_model=True)
#     res = test_multi(env, args)
#     env.agents[0].load_model(dir="./model/colight", model_id=i)
#     print(res)


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
print("creating environment...")
tf_mute_warning()
args = parse_args()
if args.test_model:
    args.config_file = generate_test_config(args.config_file)

env = create_colight_env(args, agent=args.base_policy)
# args.action_interval = 10
# args.load_dir = "/newNAS/Workspaces/DRLGroup/marko/mb-tlc/model/iteration/best/bc_colight_exp36_0"
# args.load_dir = "/newNAS/Workspaces/DRLGroup/marko/mb-tlc/model/colight/real341_Colight_20200511-165931"
# args.model_id = 1112
env.agents[0].load_model(dir=args.load_dir, model_id=args.model_id)

print("testing...")
result = test_multi(env, args)


print("Final Travel Time is %.4f" % result)