# import gym
# import argparse
# import os
# import numpy as np
# import logging
# import pickle
# from datetime import datetime
#
# from environment import TSCEnv
# from world import World
# from generator import LaneVehicleGenerator, PressurePhaseGenerator
# from metric import TravelTimeMetric
#
# from agent import MaxPressureAgent, BehaviorCloningAgent
# from agent.bc_colight_agent import BCCoLightAgent
from utils.arguments import parse_args

from utils.trajectories_buffer import TrajectoryBuffer, DoubleTrajectoryBuffer

from utils.rollout_controller import RolloutControllerMulti, RolloutControllerParallel
from utils.gail_trainer import train_bc_multi, test_multi, train_bc_share_weights
# from utils.util import build_int_intersection_map
import logging
import time
# import json

def iterations(args, env, rollout_controller, model_dirs, start_from=1, load_before=1):
    # create expert trajectory buffer
    buffer = TrajectoryBuffer if not args.new_buffer else DoubleTrajectoryBuffer
    exp_traj_buf_list = [buffer(10000, file_name="{}_{}_{}".format(args.prefix, args.base_policy, i)) for i in range(len(env.agents))]


    # create rollout controller
    rollout = rollout_controller
    if args.parameter_sharing:
        env.agents[0].save_model(step=start_from - 1, dir=model_dirs[0])
    else:
        for agent_id, agent in enumerate(env.agents):
            agent.save_model(step=start_from - 1, dir=model_dirs[agent_id])
    print("Starting iteration")
    last_result = test_multi(env, args)
    print("initial result:{}".format(last_result))
    result_history = []
    result_history.append(last_result)
    logging.info("initial result:{}".format(last_result))

    for iter in range(start_from, args.max_iterations + 1):
        t0 = time.time()
        print("iteration {}/{}".format(iter, args.max_iterations))
        logging.info("iteration {}/{}".format(iter, args.max_iterations))
        for buf_id, buf in enumerate(exp_traj_buf_list):
            buf.set_file(file_name=args.prefix+"_bc_{}".format(buf_id, iter))

        for buf in exp_traj_buf_list:
            buf.clear()
        if iter < load_before:
            pass
        elif iter == start_from and args.head_start:
            if args.head_start and iter == start_from:
                if args.base_policy == "bc_colight":
                    exp_traj_buf_list[0].set_file(file_name=HEAD_START_TRAJ_NAME)
                    exp_traj_buf_list[0].load_from_file()
                else:
                    for buf_id, buf in enumerate(exp_traj_buf_list):
                        buf.set_file(file_name="{}_{}".format(HEAD_START_TRAJ_NAME, buf_id))
                    for buf in exp_traj_buf_list:
                        buf.load_from_file()
        else:
            for i in range(0, args.walks_per_iter):
                print("performing rollout {}/{}".format(i+1, args.walks_per_iter))
                logging.info("performing rollout {}/{}".format(i+1, args.walks_per_iter))
                rollout.perform_rollout(trj_buffer_list=exp_traj_buf_list, random_walk=True, walk_rate=i*0.05, model_dirs=model_dirs, model_id=iter-1, verbose=300)
            for buf in exp_traj_buf_list:
                buf.save_to_file()

        print("rollout finished, training imitation agent")
        logging.info("rollout finished, training imitation agent")

        if iter >= load_before:
            for agent_id in range(len(env.agents)):
                env.agents[agent_id].reset()
            # if iter == 1:
            #     env.agents[0].load_model(dir=model_dirs[0], model_id=iter)
            train_bc_share_weights(args, env, exp_traj_buf_list, best_model_dirs=model_dirs, iteration_id=iter, get_best_model=True)
        else:
            if args.parameter_sharing:
                env.agents[0].load_model(dir=model_dirs[0], model_id=iter)
            else:
                for agent_id in range(len(env.agents)):
                    env.agents[agent_id].load_model(dir=model_dirs[agent_id], model_id=iter)
        # env.agents[0].load_model(dir=args.best_model_dir, model_id=iter)

        current_result = test_multi(env, args, model_dirs=model_dirs, model_id=iter)
        result_history.append(current_result)

        improvement = last_result - current_result

        if improvement < 0:
            print("negative improvement! Loading last model")
            if args.parameter_sharing:
                env.agents[0].load_model(dir=model_dirs[0], model_id=iter)
            else:
                for agent_id in range(len(env.agents)):
                    env.agents[agent_id].load_model(dir=model_dirs[agent_id], model_id=iter)
            improvement = 0
        t1 = time.time()
        print("{}: iteration {}/{} finished, current result:{:.2f}, improvement: {:.2f}, time:{}".format(args.prefix, iter, args.max_iterations,
                                                                                            current_result,
                                                                                            improvement, t1-t0))
        print(result_history)
        logging.info("iteration {}/{} finished, current result:{:.2f}, improvement: {:.2f}, time:{}".format(iter, args.max_iterations,
                                                                                            current_result,
                                                                                            improvement, t1-t0))

        last_result = current_result

#
if __name__ == '__main__':

    # parse args
    # CUDA_VISIBLE_DEVICES = [1]
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG,
        filename="./log/{}.log".format(args.prefix),
        filemode='w',
    )


    # create env
    rollout_controller = RolloutControllerParallel(args, base_policy=args.base_policy, rollout_agent_num=args.rollout_agent_num)
    env = rollout_controller.get_env()
    model_dirs = ["{}_{}_{}".format(args.best_model_dir, args.prefix, i) for i in range(len(env.agents))]

    # HEAD_START_TRAJ_NAME = "2X2_headstart_max_pressure_bc_colight" if len(env.world.intersections) < 10 else "4X4_headstart_max_pressure_bc_colight"

    # HEAD_START_TRAJ_NAME = "2X2_headstart_colight_rollout"

    HEAD_START_TRAJ_NAME = args.head_start_traj_name
    # HEAD_START_TRAJ_NAME = "4X4_headstart_colight_rollout"

    iterations(args, env, rollout_controller, model_dirs, start_from=args.start_from, load_before=args.load_before)
