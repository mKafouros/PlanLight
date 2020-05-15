import json
import argparse

def dis(a, b):   # distance between two nodes[x1, y1], [x2, y2]
    if a[0] == b[0]:
        return abs(a[1] - b[1])
    elif a[1] == b[1]:
        return abs(a[0] - b[0])
    else:
        print("warning")
        print(a)
        print(b)

def road_time(roads, road_id):
    road = roads[road_id]
    pos1 = [road["points"][0]["x"], road["points"][0]["y"]]
    pos2 = [road["points"][-1]["x"], road["points"][-1]["y"]]

    distance = dis(pos1, pos2)
    speed = road["lanes"][0]["maxSpeed"]

    return distance/speed

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

    roads_list = roadnet["roads"]
    roads = {}
    for road in roads_list:
        roads[road["id"]] = road

    total_time = 0.
    total_cnt = 0.
    for vehicle in flow:
        cost_time = 0.
        route = vehicle["route"]
        start_time = vehicle["startTime"]
        for road in route:
            cost_time += road_time(roads, road)
            # print(cost_time)
        if (cost_time+start_time < 3600):
            total_time += cost_time
            total_cnt += 1
        else:
            total_time += 3600 - start_time
            total_cnt += 1

    print((total_time) / total_cnt)
    print(total_cnt)
