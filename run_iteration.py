print("loading packages...")
from utils.arguments import parse_args

from utils.trajectories_buffer import TrajectoryBuffer, DoubleTrajectoryBuffer

from utils.rollout_controller import RolloutControllerMulti, RolloutControllerParallel
from mute_tf_warnings import tf_mute_warning
from utils.gail_trainer import train_bc_multi, test_multi, train_bc_share_weights
from utils.util import build_int_intersection_map
import logging
import time
import json

# model_path_22 = "./model/colight/syn22_Colight__0.001_0.8_0.9995_64_1000_5000_20200415-202436_colight"
# model_path_44 = "./model/colight/real44_Colight__0.001_0.8_0.9995_64_1000_5000_20200415-202306_colight"
# id22 = 481
# id44 = 191

# parse args
args = parse_args()
def rollout(args):
    MODEL_PATH = args.load_dir
    MODEL_ID = args.model_id
    tf_mute_warning()
    traj_buffer_list = [TrajectoryBuffer(10000, file_name="{}_headstart_{}_rollout".format(args.prefix, args.base_policy))]
    print("creating rollout controller...")
    rollout_controller = RolloutControllerParallel(args, base_policy=args.base_policy, rollout_agent_num=-1)
    # model_path, model_id = (model_path_22, id22) if rollout_controller.n_intersections < 10 else (model_path_44, id44)

    if not args.base_policy == "max_pressure":
        if args.base_policy == "frap":
            for agent in rollout_controller.env.agents:
                agent.load_model(dir=MODEL_PATH)
        elif args.parameter_sharing:
            rollout_controller.env.agents[0].load_model(dir=MODEL_PATH, model_id=MODEL_ID)
        else:
            raise NotImplementedError


    last_result = test_multi(rollout_controller.get_env(), args)
    print("initial result:{}".format(last_result))
    for i in range(0, args.walks_per_iter):
        print("performing rollout {}/{}".format(i+1, args.walks_per_iter))
        rollout_controller.perform_rollout(trj_buffer_list=traj_buffer_list, random_walk=True, walk_rate=i*0.05, model_dirs=[MODEL_PATH], model_id=MODEL_ID, save_traj=True, verbose=1)
        for buf in traj_buffer_list:
            buf.save_to_file()
            print("successfully saved trajectories fo rollout {}".format(i+1))

if __name__ == "__main__":
    # configs = ["config11syn1600", "config11syn2400", "config1-1", "config2-1", "config3-1", "config4-1", "config5-1"]
    configs = ["config5-1"]
    models = ["dqn", "frap", "press"]
    if args.config_file in models:
        model = args.config_file
        for config in configs:
            print(config)
            args.config_file = "config/" + config + ".json"
            args.load_dir = "./model-exp/single/{}/{}".format(model, config)
            args.base_policy = model
            args.prefix = config
            rollout(args, )

    else:
        rollout(args)

