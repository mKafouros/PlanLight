import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="run experiment")

    ''' Simulation Arguments '''
    parser.add_argument('config_file', type=str, help='path of config file')
    parser.add_argument('--thread', type=int, default=1, help='number of threads')
    parser.add_argument('--steps', type=int, default=3600, help='number of steps')
    parser.add_argument('--action_interval', type=int, default=20, help='how often agent make decisions')

    ''' Experiment Arguments'''
    parser.add_argument("--prefix", type=str, default="real341", help="")
    parser.add_argument("--max_iterations", type=int, default=200, help="max number of imitation-rollout iterations")
    parser.add_argument("--start_from", type=int, default=1, help="")
    parser.add_argument("--load_before", type=int, default=1, help="")

    ''' Path Arguments '''
    parser.add_argument("--model_dir", type=str, default="model/iteration/gail", help="directory in which model should be saved")
    parser.add_argument("--best_model_dir", type=str, default="model/iteration/best/gail", help="directory in which best model should be saved")
    parser.add_argument('--log_dir', type=str, default="log/colight", help='directory in which logs should be saved')
    parser.add_argument("--head_start_traj_name", type=str, default="head_start", help='the path of expert trajectories(if head_start is True)')

    ''' RollOut Arguments '''
    parser.add_argument("--random_walk", type=int, default=1, help="")
    parser.add_argument("--walks_per_iter", type=int, default=1, help="")
    parser.add_argument("--head_start", type=int, default=0, help="use max pressure or colight as the first base policy, need pre-saved trajectories")
    parser.add_argument("--rollout_agent_num", type=int, default=0, help="how many agents to rollout in one iteration, -1 to rollout all agents")
    parser.add_argument("--rollout_neighbor", type=int, default=0, help="choose agent to rollout randomly according to 'rollout_agent_num' or only rollout neighbours")
    parser.add_argument("--time_horizon", type=int, default=0, help="")
    parser.add_argument("--new_buffer", type=int, default=0, help="")

    ''' Training Agent Arguments'''
    parser.add_argument("--base_policy", type=str, default="bc", help="")
    parser.add_argument("--parameter_sharing", type=int, default=0, help="it has to be 1 using running colight or bc_colight")
    parser.add_argument("--cold_start", type=int, default=0, help="use max pressure as the first base policy")
    parser.add_argument("--test_model_freq", type=int, default=10, help="test model after how many training epochs")
    parser.add_argument("--episodes", type=int, default=3000, help="training episodes for one imitation")
    parser.add_argument("--batch_size", type=int, default=64, help="training batch size for imitation")

    ''' Colight Arguments'''
    parser.add_argument('--ngpu', type=str, default="0", help='gpu to be used')  # choose gpu card
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help="learning rate")
    parser.add_argument('-ls', '--learning_start', type=int, default=1000, help="learning start")
    parser.add_argument('-rs', '--replay_buffer_size', type=int, default=5000, help="size of replay buffer")
    parser.add_argument('-uf', '--update_target_model_freq', type=int, default=10, help="the frequency to update target q model")
    parser.add_argument('-gc', '--grad_clip', type=float, default=5.0, help="clip gradients")
    parser.add_argument('-ep', '--epsilon', type=float, default=0.8, help="exploration rate")
    parser.add_argument('-ed', '--epsilon_decay', type=float, default=0.9995, help="decay rate of exploration rate")
    parser.add_argument('-me', '--min_epsilon', type=float, default=0.01, help="the minimum epsilon when decaying")
    parser.add_argument('--vehicle_max', type=int, default=1, help='used to normalize node observayion')
    parser.add_argument('--mask_type', type=int, default=0, help='used to specify the type of softmax')
    parser.add_argument('--test_steps', type=int, default=3600, help='number of steps for step')
    parser.add_argument('--load_model_dir', type=str, default=None, help='load this model to test')
    parser.add_argument('--train_model', action="store_false", default=True)
    parser.add_argument('--test_model', action="store_true", default=False)
    parser.add_argument('--save_model', action="store_false", default=True)
    parser.add_argument('--load_model', action="store_true", default=False)
    parser.add_argument("--save_rate", type=int, default=40, help="save model once every time this many episodes are completed")
    parser.add_argument('--get_attention', action="store_true", default=False)
    parser.add_argument('--test_when_train', action="store_false", default=True)
    parser.add_argument('--save_dir',type=str,default="model/colight", help='directory in which model should be saved')
    parser.add_argument('--load_dir', type=str, default="model/colight", help='directory in which model should be loaded')
    parser.add_argument('--model_id', type=int, default=0, help='')

    args = parser.parse_args()

    if args.base_policy == "colight" or args.base_policy == "bc_colight":
        if not args.parameter_sharing:
            print("Warning! Forcibly setting parameter_sharing to True since base policy is colight")
            args.parameter_sharing = 1

    return args
