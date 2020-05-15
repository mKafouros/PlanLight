import json
import argparse
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="run experiment")
    parser.add_argument('config_file', type=str, help='path of config file')
    args = parser.parse_args()

    config_path = args.config_file
    config_file = open(args.config_file)
    config = json.load(config_file)

    flow_path = config["flowFile"]
    print(config)
    roadnet_path = config["roadnetFile"]
    print(flow_path)
    roadnet = json.load(open(roadnet_path))
    flow = json.load(open(flow_path))
    vehicle_stat = [0 for i in range(10)]

    for vehicle in flow:
        cost_time = 0.
        route = vehicle["route"]
        start_time = vehicle["startTime"]
        vehicle_stat[int(start_time/360)] += 10


    vehicle_stat = np.asarray(vehicle_stat)
    print(np.mean(vehicle_stat))
    print(np.std(vehicle_stat))
    print(np.max(vehicle_stat))
    print(np.min(vehicle_stat))

