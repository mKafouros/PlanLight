import numpy as np
import random
import pickle
import os
import sys

class TrajectoryBuffer(object):
    def __init__(self, size=10000, save_dir="./trajs-exp/", load_dir="./trajs-exp/", file_name="unknown"):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """

        self._storage = []
        self._maxsize = int(size)
        self._next_idx = 0

        self.save_dir = save_dir
        self.load_dir = load_dir

        self.set_file(save_dir, load_dir, file_name)

    def __len__(self):
        return len(self._storage)

    def set_file(self, save_dir=None, load_dir=None, file_name="unknown"):
        save_dir = self.save_dir if save_dir is None else save_dir
        load_dir = self.load_dir if load_dir is None else load_dir
        self.save_file = os.path.join(save_dir, file_name + ".pkl")
        self.load_file = os.path.join(load_dir, file_name + ".pkl")

    def clear(self):
        self._storage = []
        self._next_idx = 0

    def add(self, ob, action, save=False, target=None):
        data = (ob, action)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

        if save:
            self.save_to_file()

    def _encode_sample(self, idxes):
        obs, actions = [], []
        for i in idxes:
            data = self._storage[i]
            ob, action = data
            obs.append(np.array(ob, copy=False))
            actions.append(np.array(action, copy=False))
        return np.array(obs), np.array(actions)

    def make_index(self, batch_size):
        return [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]

    def make_latest_index(self, batch_size):
        idx = [(self._next_idx - 1 - i) % self._maxsize for i in range(batch_size)]
        np.random.shuffle(idx)
        return idx

    def sample_index(self, idxes):
        return self._encode_sample(idxes)

    def sample(self, batch_size, new_ratio=0):
        # Sample a batch of trajectories.
        if batch_size > 0:
            idxes = self.make_index(batch_size)
        else:
            idxes = range(0, len(self._storage))
        return self._encode_sample(idxes)

    def collect(self):
        return self.sample(-1)

    def save_to_file(self):
        data = [self._storage, self._next_idx, self._maxsize]
        # if os.path.exists(self.save_file):
        #     os.rmdir(self.save_file)
        output = open(self.save_file, 'wb')
        pickle.dump(data, output, -1)

    def load_from_file(self, mode="load"):
        if mode == "load":
            pkl_file = open(self.load_file, 'rb')
            data = pickle.load(pkl_file)
            self._storage, self._next_idx, self._maxsize = data
        elif mode == "add":
            pkl_file = open(self.load_file, 'rb')
            data = pickle.load(pkl_file)
            storage, next_idx, maxsize = data
            i = 0
            while i < next_idx:
                obs, actions = storage[i]
                self.add(obs, actions)
                i += 1


class DoubleTrajectoryBuffer(object):
    def __init__(self, size=10000, save_dir="./data/trajectories/", load_dir="./data/trajectories/", file_name="unknown"):
        self.old_buffer = TrajectoryBuffer(size=size, save_dir=save_dir, load_dir=load_dir, file_name=file_name)
        self.new_buffer = TrajectoryBuffer(size=size, save_dir=save_dir, load_dir=load_dir, file_name=file_name)

    def __len__(self):
        print("old: {}, new:{}".format(len(self.old_buffer), len(self.new_buffer)))
        return len(self.old_buffer) + len(self.new_buffer)

    def set_file(self, save_dir="./data/trajectories/", load_dir="./data/trajectories/", file_name="unknown"):
        self.old_buffer.set_file(save_dir=save_dir, load_dir=load_dir, file_name=file_name)
        self.new_buffer.set_file(save_dir=save_dir, load_dir=load_dir, file_name=file_name)

    def clear(self):
        self.old_buffer.clear()
        self.new_buffer.clear()

    def add(self, ob, action, save=False, target="old"):
        if target=="old":
            self.old_buffer.add(ob=ob, action=action, save=save)
        elif target=="new":
            self.new_buffer.add(ob=ob, action=action, save=save)
        else:
            raise Exception

    def sample(self, batch_size, new_ratio=0):
        new_samples = batch_size * new_ratio
        old_samples = batch_size - new_samples

        if old_samples > 0:
            old_obs, old_acts = self.old_buffer.sample(old_samples)
        else:
            old_obs, old_acts = None, None

        if new_samples > 0:
            new_obs, new_acts = self.new_buffer.sample(new_samples)
        else:
            new_obs, new_acts = None, None

        if old_obs is not None and new_obs is not None:
            obs = np.concatenate((old_obs, new_obs))
            acts = np.concatenate((old_acts, new_acts))
        elif old_obs is None:
            obs, acts = new_obs, new_acts
        elif new_obs is None:
            obs, acts = old_obs, old_acts
        else:
            obs, acts = None, None
        return obs, acts



    def save_to_file(self):
        self.old_buffer.save_to_file()
        self.new_buffer.save_to_file()

    def load_from_file(self, mode="load"):
        self.old_buffer.load_from_file(mode)
        self.new_buffer.load_from_file(mode)