import sys
sys.path.append("../")

import gym
from agent import *
from world import World
from environment import TSCEnv
from generator.lane_vehicle import LaneVehicleGenerator
from metric.travel_time import TravelTimeMetric
import logging
import json
import numpy as np
import random

def generate_test_config(config_path, delete_rate=0.95):
    config = json.load(open(config_path))
    flow_path = config["flowFile"]
    # print(flow_path)
    flow = json.load(open(flow_path))
    # print(len(flow))
    new_flow = []
    new_flow_path = flow_path[:-5] + "-test.json"
    new_config_path = config_path[:-5] + "-test.json"
    # print(new_flow_path)
    for i in flow:
        if delete_rate > random.random():
            new_flow.append(i)
    # print(len(new_flow))
    # print(config)
    config["flowFile"] = new_flow_path
    # print(config)

    with open(new_flow_path, 'w') as f:
        json.dump(new_flow, f)
    with open(new_config_path, 'w') as f:
        json.dump(config, f)

    return new_config_path


def build_int_intersection_map(roadnet_file, node_degree=4):
    '''
    generate the map between int ---> intersection ,intersection --->int
    generate the map between int ---> roads,  roads ----> int
    generate the required adjacent matrix
    generate the degree vector of node (we only care the valid roads(not connect with virtual intersection), and intersections)
    return: map_dict, and adjacent matrix
    save res into save dir, the res is as below
    res = [net_node_dict_id2inter,net_node_dict_inter2id,net_edge_dict_id2edge,net_edge_dict_edge2id,
        node_degree_node,node_degree_edge,node_adjacent_node_matrix,node_adjacent_edge_matrix,
        edge_adjacent_node_matrix]
    '''

    def get_road_dict(roadnet_dict, road_id):
        for item in roadnet_dict['roads']:
            if item['id'] == road_id:
                return item
        print("Cannot find the road id {0}".format(road_id))
        sys.exit(-1)
    roadnet_dict = json.load(open(roadnet_file, "r"))
    valid_intersection_id = [i["id"] for i in roadnet_dict["intersections"] if not i["virtual"]]
    net_node_dict_id2inter = {}
    net_node_dict_inter2id = {}
    net_edge_dict_id2edge = {}
    net_edge_dict_edge2id = {}
    node_degree_node = []  # the num of adjacent nodes of node
    node_degree_edge = []  # the valid num of adjacent edges of node
    node_adjacent_node_matrix = []  # adjacent node of each node
    node_adjacent_edge_matrix = []  # adjacent edge of each node
    edge_adjacent_node_matrix = []  # adjacent node of each edge
    invalid_roads = []
    cur_num = 0
    # build the map between id and intersection
    for node_dict in roadnet_dict["intersections"]:
        if node_dict["virtual"]:
            for i in node_dict["roads"]:
                invalid_roads.append(i)
            continue
        node_id = node_dict["id"]
        net_node_dict_id2inter[cur_num] = node_id
        net_node_dict_inter2id[node_id] = cur_num
        cur_num += 1
    # map between id and intersection built done
    if cur_num != len(valid_intersection_id):
        print("cur_num={}".format(cur_num))
        print("valid_intersection_id length={}".format(len(valid_intersection_id)))
        raise ValueError("cur_num should equal to len(valid_intersection_id)")

    cur_num = 0
    for edge_dict in roadnet_dict["roads"]:
        edge_id = edge_dict["id"]
        if edge_id in invalid_roads:
            continue
        else:
            net_edge_dict_id2edge[cur_num] = edge_id
            net_edge_dict_edge2id[edge_id] = cur_num
            cur_num += 1
            input_node = edge_dict['startIntersection']
            output_node = edge_dict['endIntersection']
            input_node_id = net_node_dict_inter2id[input_node]
            output_node_id = net_node_dict_inter2id[output_node]
            edge_adjacent_node_matrix.append([input_node_id, output_node_id])

    # build adjacent matrix for node (i.e the adjacent node of the node, and the
    # adjacent edge of the node)
    for node_dict in roadnet_dict["intersections"]:
        if node_dict["virtual"]:
            continue
        node_id = node_dict["id"]
        road_links = node_dict['roads']
        input_nodes = []  # should be node_degree
        input_edges = []  # needed, should be node_degree
        for road_link_id in road_links:
            road_link_dict = get_road_dict(roadnet_dict, road_link_id)
            if road_link_dict['endIntersection'] == node_id:
                if road_link_id in net_edge_dict_edge2id.keys():
                    input_edge_id = net_edge_dict_edge2id[road_link_id]
                    input_edges.append(input_edge_id)
                else:
                    continue
                start_node = road_link_dict['startIntersection']
                if start_node in net_node_dict_inter2id.keys():
                    start_node_id = net_node_dict_inter2id[start_node]
                    input_nodes.append(start_node_id)
        if len(input_nodes) != len(input_edges):
            print(len(input_nodes))
            print(len(input_edges))
            print(node_id)
            raise ValueError("len(input_nodes) should be equal to len(input_edges)")
        node_degree_node.append(len(input_nodes))
        node_degree_edge.append(len(input_edges))
        while len(input_nodes) < node_degree:
            input_nodes.append(0)
        while len(input_edges) < node_degree:
            input_edges.append(0)
        node_adjacent_edge_matrix.append(input_edges)
        node_adjacent_node_matrix.append(input_nodes)

    node_degree_node = np.array(node_degree_node)  # the num of adjacent nodes of node
    node_degree_edge = np.array(node_degree_edge)  # the valid num of adjacent edges of node
    node_adjacent_node_matrix = np.array(node_adjacent_node_matrix)
    node_adjacent_edge_matrix = np.array(node_adjacent_edge_matrix)
    edge_adjacent_node_matrix = np.array(edge_adjacent_node_matrix)

    res = [net_node_dict_id2inter, net_node_dict_inter2id, net_edge_dict_id2edge, net_edge_dict_edge2id,
           node_degree_node, node_degree_edge, node_adjacent_node_matrix, node_adjacent_edge_matrix,
           edge_adjacent_node_matrix]
    return res



def argmin(arr):
    min = arr[0]
    mini = 0
    for i in range(1, len(arr)):
        if arr[i] < min:
            min = arr[i]
            mini = i
    return mini

