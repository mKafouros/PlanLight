import gym
from environment import TSCEnv
from world import World
from generator import LaneVehicleGenerator
from agent import SOTLAgent
from metric import TravelTimeMetric
from utils.arguments import parse_args
from utils.util import generate_test_config


def run(args):
    # create world
    world = World(args.config_file, thread_num=args.thread, silent=True)

    # create agents
    agents = []
    for i in world.intersections:
        action_space = gym.spaces.Discrete(len(i.phases))
        agents.append(SOTLAgent(
            action_space, i, world,
            LaneVehicleGenerator(world, i, ["lane_count"], in_only=True, average=None),
            LaneVehicleGenerator(world, i, ["lane_waiting_count"], in_only=True, average="all", negative=True)
        ))

    # create metric
    metric = TravelTimeMetric(world)

    # create env
    env = TSCEnv(world, agents, metric)

    # simulate
    obs = env.reset()
    actions = []
    i = 0
    reward_sum = 0.
    reward_cnt = 0.
    while i < args.steps:
        if i % args.action_interval == 0:
            actions = []
            for agent_id, agent in enumerate(agents):
                actions.append(agent.get_action(obs[agent_id]))
            for _ in range(args.action_interval):
                obs, rewards, dones, info = env.step(actions)
                i += 1
            for reward in rewards:
                reward_sum += reward
                reward_cnt += 1
    print("avg queue length: {}".format(reward_sum / reward_cnt))

    result = env.eng.get_average_travel_time()
    return result

if __name__ == '__main__':
    args = parse_args()
    if args.config_file == "test":
        configs = ["config11syn1600", "config11syn2400", "config1-1", "config2-1", "config3-1", "config4-1",
                   "config5-1"]
        for config in configs:
            result_sum = 0.
            print(config)
            for i in range(10):
                args.config_file = "config/" + config + ".json"
                args.config_file = generate_test_config(args.config_file)
                result_sum += run(args)
            print("Average Travel Time is %.4f" % (result_sum / 10))
    if args.config_file == "exp":
        configs = ["config22syn", "config33mysyn", "config66mysyn", "config44-real3", "config34-real"]
        for config in configs:
            args.config_file = "config/" + config + ".json"
            result_sum = 0.
            print(config)
            result = run(args)
            print("Average Travel Time is %.4f" % result)
    else:
        result = run(args)
        print("Final Travel Time is %.4f" % result)