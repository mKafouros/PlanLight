import os
import numpy as np
import logging
import time
'''Currently only support single agent.''' #TODO multi-agent version
def train_gail(args, env, exp_traj_buf, iteration_id=0, get_best_model=True):
    total_decision_num = 0.
    sum_episodes_time_10 = 0
    best_episode_time = 1e5
    for e in range(1, args.episodes + 1):
        last_obs = env.reset()

        history_actions, history_obs, history_v_preds, history_v_preds_next = [  [ []for i in range(len(env.agents)) ] for i in range(4)  ]

        episodes_decision_num = 0.
        step = 0
        while step < args.steps:
            if step % args.action_interval == 0:
                actions = []
                for agent_id, agent in enumerate(env.agents):
                    action, v_pred = agent._act(last_obs[agent_id], stochastic=False)
                    actions.append(action)

                    history_obs[agent_id].append(last_obs[agent_id])
                    history_actions[agent_id].append(action)
                    history_v_preds[agent_id].append(v_pred)

                for _ in range(args.action_interval):
                    obs, rewards, dones, _ = env.step(actions)
                    step += 1

                episodes_decision_num += 1
                total_decision_num += 1

                last_obs = obs
            # print(history_actions)

            if len(history_v_preds[0]) >= args.batch_size:
                expert_obs, expert_actions = exp_traj_buf.sample(args.batch_size)
                for agent_id, agent in enumerate(env.agents):
                    agent.update_discriminator(expert_obs, expert_actions, history_obs[agent_id], history_actions[agent_id])
                    _, v_pred = agent._act(obs[agent_id])
                    history_v_preds_next[agent_id] = history_v_preds[agent_id][1:] + [np.asscalar(v_pred)]
                    agent.update_policy(history_obs[agent_id], history_actions[agent_id], history_v_preds[agent_id], history_v_preds_next[agent_id], batch_size=args.batch_size)
                    history_actions, history_obs, history_v_preds, history_v_preds_next = [  [ [] for i in range(len(env.agents)) ] for i in range(4)  ]

            if all(dones):
                break
        if e % args.save_rate == args.save_rate - 1:
            for agent in env.agents:
                agent.save_model(e, args.model_dir)
                # print("model saved")
        episode_time = test(env, args)
        if episode_time < best_episode_time:
            best_episode_time = episode_time
            for agent in env.agents:
                agent.save_model(step=iteration_id, dir=args.best_model_dir)
                print("best model saved, episode: {}, result: {:.2f}".format(e, best_episode_time))

        sum_episodes_time_10 += episode_time
        # print("episode:{}/{}, average travel time:{}".format(e, args.episodes, env.eng.get_average_travel_time()))
        # test_single(env.agents[0], env)
        # for agent_id, agent in enumerate(env.agents):
        #     print("agent:{}, mean_episode_reward:{}".format(agent_id, episodes_rewards[agent_id] / episodes_decision_num))

        if e % 10 == 0:
            print("episode:{}/{}, average travel time:{:.2f}".format(e, args.episodes, sum_episodes_time_10/10.))
            sum_episodes_time_10 = 0
    if get_best_model:
        env.agents[0].load_model(dir=args.best_model_dir, model_id=iteration_id)

    return args.best_model_dir + "-" + str(iteration_id)



'''Currently only support single agent.'''  # TODO multi-agent version
def train_bc(args, env, exp_traj_buf, iteration_id=0, get_best_model=True):
    sum_episodes_time = 0.
    sum_episodes_time_cnt = 0.
    best_episode_time = 1e5
    sum_loss_cnt = 0.
    sum_loss = 0.
    for e in range(1, args.episodes + 1):
        obs, actions = exp_traj_buf.sample(args.batch_size)
        for i in range(5):
            loss = env.agents[0].update_policy(obs, actions)
            sum_loss_cnt += 1
            sum_loss += loss
        # if e % args.save_rate == args.save_rate - 1 and args.save_model:
        #     if not os.path.exists(args.save_dir):
        #         os.makedirs(args.save_dir)
        #     for agent in env.agents:
        #         agent.save_model(e, args.save_dir)
        #         # print("model saved")
        if e % 10 == 0:
            episode_time = test(env, args)
            if episode_time < best_episode_time:
                best_episode_time = episode_time
                for agent in env.agents:
                    agent.save_model(step=iteration_id, dir=args.save_dir)
                    print("best model saved, episode: {}, result: {:.2f}".format(e, best_episode_time))

            sum_episodes_time += episode_time
            sum_episodes_time_cnt += 1
        # print("episode:{}/{}, average travel time:{}".format(e, args.episodes, env.eng.get_average_travel_time()))
        # test_single(env.agents[0], env)
        # for agent_id, agent in enumerate(env.agents):
        #     print("agent:{}, mean_episode_reward:{}".format(agent_id, episodes_rewards[agent_id] / episodes_decision_num))

        if e % 10 == 0:
            print("episode:{}/{}, average travel time:{:.2f}, loss: {}".format(e, args.episodes, sum_episodes_time / sum_episodes_time_cnt, sum_loss / sum_loss_cnt))
            sum_episodes_time = 0
            sum_episodes_time_cnt = 0
            sum_loss_cnt = 0.
            sum_loss = 0.

    if get_best_model:
        env.agents[0].load_model(dir=args.save_dir, model_id=iteration_id)

def train_bc_multi(args, env, exp_traj_buf_list, best_model_dirs, iteration_id=0, get_best_model=True):
    total_decision_num = 0.
    sum_episodes_time_10 = 0
    best_episode_time = None
    for e in range(1, args.episodes + 1):
        # obs, actions = [], []
        for i in range(len(env.agents)):
            ob, action = exp_traj_buf_list[i].sample(args.batch_size)
            env.agents[i].update_policy(ob, action)
        episode_time = test(env, args)
        if best_episode_time is None or episode_time < best_episode_time:
            best_episode_time = episode_time
            for agent_id, agent in enumerate(env.agents):
                agent.save_model(step=iteration_id, dir=best_model_dirs[agent_id])
            print("best model saved, episode: {}, result: {:.2f}".format(e, best_episode_time))
            logging.info("best model saved, episode: {}, result: {:.2f}".format(e, best_episode_time))

        sum_episodes_time_10 += episode_time

        if e % 10 == 0:
            print("episode:{}/{}, average travel time:{:.2f}".format(e, args.episodes, sum_episodes_time_10 / 10.))
            logging.info("episode:{}/{}, average travel time:{:.2f}".format(e, args.episodes, sum_episodes_time_10 / 10.))
            sum_episodes_time_10 = 0

    if get_best_model:
        for agent_id, agent in enumerate(env.agents):
            agent.load_model(dir=best_model_dirs[agent_id], model_id=iteration_id)

def train_bc_share_weights(args, env, exp_traj_buf_list, best_model_dirs, iteration_id=0, get_best_model=True):
    # sum_episodes_time = 0
    best_episode_time = None
    sum_loss = 0
    sum_loss_cnt = 0
    best_loss = None
    last_result = None
    for e in range(1, args.episodes + 1):
        t0 = time.time()
        for _ in range(20):
            obs, actions = exp_traj_buf_list[0].sample(args.batch_size, new_ratio=0.5)
            permutation = np.random.permutation(range(len(obs)))
            obs = obs[permutation]
            actions = actions[permutation]
            loss = env.agents[0].update_policy(obs, actions)
            sum_loss_cnt += 1
            sum_loss += loss
        t1 = time.time()
        if e > args.episodes / 50. or e == 1:
            if e == 1 :
                env.agents[0].save_model(step=iteration_id, dir=best_model_dirs[0])
            if e % args.test_model_freq == 0:
                episode_time = test_multi(env, args, model_dirs=best_model_dirs, model_id=iteration_id)
                last_result = episode_time
            #     sum_episodes_time += episode_time
            elif best_loss is None or loss < best_loss:
                best_loss = loss
                episode_time = test_multi(env, args, model_dirs=best_model_dirs, model_id=iteration_id)
                last_result = episode_time
            else:
                episode_time =  None
            t2 = time.time()
            if episode_time is not None and (best_episode_time is None or episode_time < best_episode_time):
                best_episode_time = episode_time
                # for agent_id, agent in enumerate(env.agents):
                env.agents[0].save_model(step=iteration_id, dir=best_model_dirs[0])
                print("best model saved, episode: {}, result: {:.2f}".format(e, best_episode_time))
                logging.info("best model saved, episode: {}, result: {:.2f}".format(e, best_episode_time))

        if e % 10 == 0:
            print("episode:{}/{}, average loss: {}".format(e, args.episodes, sum_loss/sum_loss_cnt))
            print("last travel time: {}".format(last_result))
            logging.info("episode:{}/{}, average loss: {}".format(e, args.episodes, sum_loss/10.0))
            # print("time - train: {}, test: {}".format(t1-t0, t2-t1))
            sum_episodes_time = 0
            sum_loss = 0
            sum_loss_cnt = 0

    if get_best_model:
        env.agents[0].load_model(dir=best_model_dirs[0], model_id=iteration_id)

def test_multi(env, args, load=False, model_dirs=[], model_id=None):
    if load:
        if args.parameter_sharing:
            env.agents[0].load_model(dir=model_dirs[0], model_id=model_id)
    i = 0
    obs = env.reset()
    while i < args.steps:
        if i % args.action_interval == 0:
            actions = []
            for agent_id, agent in enumerate(env.agents):
                if args.parameter_sharing:
                    # agent = env.agents[0]
                    actions = env.agents[0].get_actions(obs)
                    break
                actions.append(agent.get_action(obs[agent_id]))
            for _ in range(args.action_interval):
                obs, rewards, dones, info = env.step(actions)
                i += 1
            # print(actions)

        if all(dones):
            break

    trv_time = env.eng.get_average_travel_time()
    return trv_time


def test(env, args, load=False, dir="model/iteration/gail", model_id=None):
    if load:
        env.agents[0].load_model(dir, model_id)
    i = 0
    obs = env.reset()
    while i < args.steps:
        if i % args.action_interval == 0:
            actions = []
            for agent_id, agent in enumerate(env.agents):
                actions.append(agent.get_action(obs[agent_id]))
            for _ in range(args.action_interval):
                obs, rewards, dones, info = env.step(actions)
                i += 1
        # print(rewards)

        if all(dones):
            break

    trv_time = env.eng.get_average_travel_time()
    # print("Final Travel Time is %.4f" % trv_time)
    return trv_time