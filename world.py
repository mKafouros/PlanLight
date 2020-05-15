import json
import os.path as osp
import os
import cityflow

import numpy as np
from math import atan2, pi
import sys
import copy
import pickle
import time

def _get_direction(road, out=True):
    if out:
        x = road["points"][1]["x"] - road["points"][0]["x"]
        y = road["points"][1]["y"] - road["points"][0]["y"]
    else:
        x = road["points"][-2]["x"] - road["points"][-1]["x"]
        y = road["points"][-2]["y"] - road["points"][-1]["y"]
    tmp = atan2(x, y)
    return tmp if tmp >= 0 else (tmp + 2*pi)

class Intersection(object):
    def __init__(self, intersection, world):
        self.id = intersection["id"]
        self.eng = world.eng
        
        # incoming and outgoing roads of each intersection, clock-wise order from North
        self.roads = []
        self.outs = []
        self.directions = []
        self.out_roads = None
        self.in_roads = None

        # links and phase information of each intersection
        self.roadlinks = []
        self.lanelinks_of_roadlink = []
        self.startlanes = []
        self.lanelinks = []
        self.phase_available_roadlinks = []
        self.phase_available_lanelinks = []
        self.phase_available_startlanes = []

        # define yellow phases, currently default to 0
        self.yellow_phase_id = [0]
        self.yellow_phase_time = 3

        # parsing links and phases
        for roadlink in intersection["roadLinks"]:
            self.roadlinks.append((roadlink["startRoad"], roadlink["endRoad"]))
            lanelinks = []
            for lanelink in roadlink["laneLinks"]:
                startlane = roadlink["startRoad"] + "_" + str(lanelink["startLaneIndex"])
                self.startlanes.append(startlane)
                endlane = roadlink["endRoad"] + "_" + str(lanelink["endLaneIndex"])
                lanelinks.append((startlane, endlane))
            self.lanelinks.extend(lanelinks)
            self.lanelinks_of_roadlink.append(lanelinks)
        # print(self.roadlinks)
        # print(self.lanelinks_of_roadlink)

        self.startlanes = list(set(self.startlanes))

        phases = intersection["trafficLight"]["lightphases"]
        self.phases = [i for i in range(len(phases)) if not i in self.yellow_phase_id]
        for i in self.phases:
            phase = phases[i]
            self.phase_available_roadlinks.append(phase["availableRoadLinks"])
            phase_available_lanelinks = []
            phase_available_startlanes = []
            for roadlink_id in phase["availableRoadLinks"]:
                lanelinks_of_roadlink = self.lanelinks_of_roadlink[roadlink_id]
                phase_available_lanelinks.extend(lanelinks_of_roadlink)
                for lanelinks in lanelinks_of_roadlink:
                    phase_available_startlanes.append(lanelinks[0])
            self.phase_available_lanelinks.append(phase_available_lanelinks)
            phase_available_startlanes = list(set(phase_available_startlanes))
            self.phase_available_startlanes.append(phase_available_startlanes)

        self.reset()


    def insert_road(self, road, out):
        self.roads.append(road)
        self.outs.append(out)
        self.directions.append(_get_direction(road, out))

    def sort_roads(self, RIGHT):
        order = sorted(range(len(self.roads)), key=lambda i: (self.directions[i], self.outs[i] if RIGHT else not self.outs[i]))
        self.roads = [self.roads[i] for i in order]
        self.directions = [self.directions[i] for i in order]
        self.outs = [self.outs[i] for i in order]
        self.out_roads = [self.roads[i] for i, x in enumerate(self.outs) if x]
        self.in_roads = [self.roads[i] for i, x in enumerate(self.outs) if not x]

    def _change_phase(self, phase, interval):
        self.eng.set_tl_phase(self.id, phase)
        self._current_phase = phase
        self.current_phase_time = interval

    def step(self, action, interval):
        # if current phase is yellow, then continue to finish the yellow phase
        # recall self._current_phase means true phase id (including yellows)
        # self.current_phase means phase id in self.phases (excluding yellow)
        # print("current_phase: {0}, _current_phase: {1}".format(self.current_phase, self._current_phase))
        if self._current_phase in self.yellow_phase_id:
            if self.current_phase_time >= self.yellow_phase_time:
                self._change_phase(self.phases[self.action_before_yellow], interval)
                self.current_phase = self.action_before_yellow
            else:
                self.current_phase_time += interval
        else:
            if action == self.current_phase:
                self.current_phase_time += interval
            else:
                if self.yellow_phase_time > 0:
                    self._change_phase(self.yellow_phase_id[0], interval)
                    self.action_before_yellow = action
                else:
                    self._change_phase(action, interval)
                    self.current_phase = action

    def take_snapshot(self):
        intersection_info = self.current_phase, self._current_phase, self.current_phase_time, self.action_before_yellow
        return intersection_info

    def load_snapshot(self, intersection_info):
        self.current_phase, self._current_phase, self.current_phase_time, self.action_before_yellow = intersection_info

    def reset(self):
        # record phase info
        self.current_phase = 0 # phase id in self.phases (excluding yellow)
        self._current_phase = self.phases[0] # true phase id (including yellow)
        self.eng.set_tl_phase(self.id, self._current_phase)
        self.current_phase_time = 0
        self.action_before_yellow = None



class World(object):
    """
    Create a CityFlow engine and maintain informations about CityFlow world
    """
    def __init__(self, cityflow_config, thread_num, max_steps=None, silent=False):
        if not silent:
            print("building world...")
        self.cityflow_config = cityflow_config
        self.thread_num = thread_num
        self.eng = cityflow.Engine(cityflow_config, thread_num=thread_num)
        with open(cityflow_config) as f:
            cityflow_config = json.load(f)
        self.roadnet = self._get_roadnet(cityflow_config)
        self.RIGHT = True # vehicles moves on the right side, currently always set to true due to CityFlow's mechanism
        self.interval = cityflow_config["interval"]

        # get all non virtual intersections
        self.intersections = [i for i in self.roadnet["intersections"] if not i["virtual"]]
        self.intersection_ids = [i["id"] for i in self.intersections]
        self.intersection_positions = np.array([list(map(int, i.split("_")[-2:])) for i in self.intersection_ids])
        # print(self.intersection_positions)
        self.intersections_dim = (np.max(self.intersection_positions[:,0]), np.max(self.intersection_positions[:,1]))
        self.intersection_map = np.zeros(self.intersections_dim)
        for i in range(1, self.intersections_dim[0] + 1):
            for j in range(1, self.intersections_dim[1] + 1):
                self.intersection_map[i - 1][j - 1] = self.intersection_positions.tolist().index([i, j])
        # print(self.intersection_map)



        # create non-virtual Intersections
        if not silent:
            print("creating intersections...")
        non_virtual_intersections = [i for i in self.roadnet["intersections"] if not i["virtual"]]
        self.intersections = [Intersection(i, self) for i in non_virtual_intersections]
        self.intersection_ids = [i["id"] for i in non_virtual_intersections]
        self.id2intersection = {i.id: i for i in self.intersections}
        if not silent:
            print("intersections created.")

        # id of all roads and lanes
        if not silent:
            print("parsing roads...")
        self.all_roads = []
        self.all_lanes = []

        for road in self.roadnet["roads"]:
            self.all_roads.append(road["id"])
            i = 0
            for _ in road["lanes"]:
                self.all_lanes.append(road["id"] + "_" + str(i))
                i += 1

            iid = road["startIntersection"]
            if iid in self.intersection_ids:
                self.id2intersection[iid].insert_road(road, True)
            iid = road["endIntersection"]
            if iid in self.intersection_ids:
                self.id2intersection[iid].insert_road(road, False)

        for i in self.intersections:
            i.sort_roads(self.RIGHT)
        if not silent:
            print("roads parsed.")

        # initializing info functions
        self.info_functions = {
            "vehicles": (lambda : self.eng.get_vehicles(include_waiting=True)),
            "lane_count": self.eng.get_lane_vehicle_count,
            "lane_waiting_count": self.eng.get_lane_waiting_vehicle_count,
            "lane_vehicles": self.eng.get_lane_vehicles,
            "time": self.eng.get_current_time,
            "pressure": self.get_pressure
        }
        self.fns = []
        self.info = {}

        self.vehicle_enter_time = {}
        self.travel_times = []
        self.done = False

        self.max_steps = max_steps
        self.steps = 0

        if not silent:
            print("world built.")

    # def _get_pressure(self):
    #     # TODO padding 0 if less than 12 roadlinks
    #     pressures = []
    #     lane_count = self.eng.get_lane_vehicle_count()
    #     for intersection in self.intersections:
    #         intersection_pressures = []
    #         lane_road_links = intersection.lanelinks_of_roadlink
    #         road_links = intersection.roadlinks
    #         pressure = 0
    #         for id, lane_links in enumerate(lane_road_links):
    #             start_road = road_links[id][0]
    #             pressure = 0
    #             for lane_link in lane_links:
    #                 pressure += (lane_count[lane_link[0]] - lane_count[lane_link[1]])
    #             pressure /= 3.
    #             intersection_pressures.append(pressure)
    #         pressures.append(pressure)
    #     return pressures

    def get_pressure(self):
        vehicles = self.eng.get_lane_vehicle_count()
        pressures = {}
        for i in self.intersections:
            pressure = 0
            in_lanes = []
            for road in i.in_roads:
                from_zero = (road["startIntersection"] == i.id) if self.RIGHT else (
                        road["endIntersection"] == i.id)
                for n in range(len(road["lanes"]))[::(1 if from_zero else -1)]:
                    in_lanes.append(road["id"] + "_" + str(n))
            out_lanes = []
            for road in i.out_roads:
                from_zero = (road["endIntersection"] == i.id) if self.RIGHT else (
                        road["startIntersection"] == i.id)
                for n in range(len(road["lanes"]))[::(1 if from_zero else -1)]:
                    out_lanes.append(road["id"] + "_" + str(n))
            for lane in vehicles.keys():
                if lane in in_lanes:
                    pressure += vehicles[lane]
                if lane in out_lanes:
                    pressure -= vehicles[lane]
            pressures[i.id] = pressure
        return pressures

    def _get_roadnet(self, cityflow_config):
        roadnet_file= osp.join(cityflow_config["dir"], cityflow_config["roadnetFile"])
        with open(roadnet_file) as f:
            roadnet = json.load(f)
        return roadnet

    def subscribe(self, fns):
        if isinstance(fns, str):
            fns = [fns]
        for fn in fns:
            if fn in self.info_functions:
                if not fn in self.fns:
                    self.fns.append(fn)
            else:
                raise Exception("info function %s not exists" % fn)

    def step(self, actions=None):
        self.steps += 1
        self.done = False
        if self.max_steps is not None and self.max_steps <= self.steps:
            self.done = True
        if actions is not None:
            for i, action in enumerate(actions):
                self.intersections[i].step(action, self.interval)
        # print("mark1")
        self.eng.next_step()
        # print("mark")
        self._update_infos()

    def take_snapshot(self, to_file=False, dir="./archive/", verbose=False, file_name="snapshot"):
        if verbose:
            print("taking snapshot...current time: {0}".format(self.eng.get_current_time()))
        archive = self.eng.snapshot()

        if not osp.exists(dir):
            os.makedirs(dir)

        intersection_infos = []
        for i in self.intersections:
            intersection_infos.append(i.take_snapshot())
        if to_file:
            archive_name = file_name + "_archive.json"
            archive_path = osp.join(dir, archive_name)

            info_name = file_name + "_info.pkl"
            info_path = osp.join(dir, info_name)
            info_file = open(info_path, "wb")

            if os.path.exists(archive_path):
                os.remove(archive_path)

            archive.dump(archive_path)
            pickle.dump(intersection_infos, info_file, protocol=pickle.HIGHEST_PROTOCOL)
            archive_file_test = open(archive_path)
            _ = json.load(archive_file_test)

            info_file.close()
        return archive, intersection_infos

    def load_snapshot(self, archive=None, from_file=False, intersection_infos=None, verbose=False, dir="./archive", file_name="snapshot"):
        if verbose:
            print("loading snapshot...current time: {0}".format(self.eng.get_current_time()))
        if archive is None and not from_file:
            return FileNotFoundError
        if from_file:
            archive_name = file_name + "_archive.json"
            archive_path = osp.join(dir, archive_name)

            info_name = file_name + "_info.pkl"
            info_path = osp.join(dir, info_name)
            info_file = open(info_path, "rb")

            archive_file_test = open(archive_path)
            archive_file_test = json.load(archive_file_test) # in order to throw a python exception when archive file is not properly saved

            self.eng.load_from_file(archive_path) # only throw c++ exception when file is not properly saved, cannot catch
            intersection_infos = pickle.load(info_file)

            info_file.close()
        else:
            self.eng.load(archive)
        for id, i in enumerate(self.intersections):
            i.load_snapshot(intersection_infos[id])
        self._update_infos()

        if verbose:
            print("snapshot loaded...current time: {0}".format(self.eng.get_current_time()))

    def reset(self):
        self.eng.reset()
        for I in self.intersections:
            I.reset()
        self._update_infos()
        self.steps = 0
        self.done = False

    def _update_infos(self):
        self.info = {}
        for fn in self.fns:
            self.info[fn] = self.info_functions[fn]()

    def get_info(self, info):
        return self.info[info]

    def debug_info(self):
        print(self.vehicle_enter_time)
        print(self.travel_times)
        print(self.eng.get_vehicles())

    def reset_eng(self, config, thread_num):
        self.eng = cityflow.Engine(config, thread_num=thread_num)


if __name__ == "__main__":
    world = World("examples/config.json", thread_num=1)
    #print(len(world.intersections[0].startlanes))
    print(world.intersections[0].phase_available_startlanes)