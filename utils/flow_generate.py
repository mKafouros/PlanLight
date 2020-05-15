import json
import argparse
import random
import numpy as np
import os

template = {
    "vehicle":
         {
             "length": 5.0,
              "width": 2.0,
              "maxPosAcc": 2.0,
              "maxNegAcc": 4.5,
              "usualPosAcc": 2.0,
              "usualNegAcc": 4.5,
              "minGap": 2.5,
              "maxSpeed": 11.111,
              "headwayTime": 2
         },
     "route": ["road_0_1_0", "road_1_1_0", "road_2_1_3"],
     "interval": 1.0,
     "startTime": 0,
     "endTime": 0
}

turning_ratio = [0.1, 0.8, 0.1]  #left, straight, right

def target_intersection_id(road):
    ids = list(map(int, road.split("_")[1:]))
    direction = ids[2]
    if direction == 0:
        ids[0] += 1
    elif direction == 1:
        ids[1] += 1
    elif direction == 2:
        ids[0] -= 1
    else:
        ids[1] -= 1

    return [ids[0], ids[1]]

def turn_to(road):
    directions = [1, 0, 3, 2]  # clockwise, start from go-north
    intersection_id = target_intersection_id(road)
    current_direction = list(map(int, road.split("_")[1:]))[-1]
    current_direction_idx = directions.index(current_direction)
    rand = random.random()
    if rand < turning_ratio[0]:  # turn left
        direction = directions[(current_direction_idx - 1) % 4]
    elif rand > turning_ratio[0] + turning_ratio[1]:  # turn right
        direction = directions[(current_direction_idx - 1) % 4]
    else:
        direction = current_direction

    if intersection_id[0] >= max_size[0] or intersection_id[1] >= max_size[1] or intersection_id[0] <= 0 or intersection_id[1] <= 0:
        return None
    else:
        return "road_{}_{}_{}".format(intersection_id[0], intersection_id[1], direction)

def get_initial_roads():
    roads = []
    for j in range(1, max_size[1]):
        roads.append("road_{}_{}_{}".format(0, j, 0))
        roads.append("road_{}_{}_{}".format(max_size[0], j, 2))

    for i in range(1, max_size[0]):
        roads.append("road_{}_{}_{}".format(i, 0, 1))
        roads.append("road_{}_{}_{}".format(i, max_size[1], 3))

    return roads

def create_vehicle(time, route):
    vehicle = template.copy()
    vehicle["startTime"] = time
    vehicle["endTime"] = time
    vehicle["route"] = route
    return vehicle

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

    global max_size
    intersections = roadnet["intersections"]
    ids = []
    for intersection in intersections:
        i_id = list(map(int, intersection["id"].split("_")[1:]))
        ids.append(np.asarray(i_id))
    ids = np.asarray(ids)
    max_size = ids.max(axis=0)

    count = {}
    for vehicle in flow:
        if vehicle["route"][0] not in count:
            count[vehicle["route"][0]] = 1
        else:
            count[vehicle["route"][0]] += 1
    print(count)

    init_roads = get_initial_roads()
    vehicles = []
    time = 0
    while time < 3600:
        for start_road in init_roads:
            route = []
            route.append(start_road)
            next_road = turn_to(start_road)
            while next_road is not None:
                route.append(next_road)
                next_road = turn_to(next_road)
            # print(route)
            vehicle = create_vehicle(time, route)
            vehicles.append(vehicle)
        time += 6
    print(len(vehicles))
    #
    # dir =

    target_data_dir = "dataset/{}X{}_mysyn/".format(max_size[0] - 1, max_size[1] - 1)

    if not os.path.exists(target_data_dir):
        os.mkdir(target_data_dir)

    new_config_path = "config/config{}{}mysyn.json".format(max_size[0] - 1, max_size[1] - 1)
    new_roadnet_path = target_data_dir + "roadnet.json"

    new_flow_path = target_data_dir + "flow.json"

    with open(new_flow_path, 'w') as f:
        json.dump(vehicles, f, indent=1)

    with open(new_roadnet_path, 'w') as f:
        json.dump(roadnet, f, indent=1)

    config["flowFile"] = new_flow_path
    config["roadnetFile"] = new_roadnet_path

    with open(new_config_path, 'w') as f:
        json.dump(config, f, indent=1)



    # vehicles = []
    #
    # vehicle = create_vehicle(5, ["road_0_1_0", "road_1_1_0"])
    # print(vehicle)
    # print(template)





