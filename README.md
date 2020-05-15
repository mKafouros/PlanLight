# PlanLight

code for paper PlanLight: Learning to Optimize Traffic Signal Control with Planning and Iterative Policy Improvement.

An RL and planning based method for TSC simulated on CityFLow.

#### How to run

train PlanLight from scratch

```
python run_iterations.py CONFIG_FILE_PATH
```

The config file is required for running CityFlow, we provided several sample configs in ./config/

Setting existing method as the first base_policy for PlanLight

1. train the model you want to use. eg. CoLight:

   ```
   python run_colight.py CONFIG_FILE_PATH --save_dir MODEL_PATH
   ```

   the first step can be ommited using non-RL-based methods as first base policy

2. run onestep rollout to collect trajectories:

   ```
   python run_iteration.py CONFIG_FILE_PATH --model_dir MODEL_PATH --base_policy colight
   ```

3. run PlanLight

   ```
   python run_iterations.py CONFIG_FILE_PATH --head_start_traj_name TRAJECTORY_FILE
   ```