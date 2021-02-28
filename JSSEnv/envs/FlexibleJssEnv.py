import random
from pathlib import Path

import gym
import datetime
import pandas as pd
import numpy as np

class FlexibleJssEnv(gym.Env):

    def __init__(self, env_config=None):
        if env_config is None:
            env_config = {'instance_path': str(Path(__file__).parent.absolute()) + '/instances/flexible/mt10c1.fjs'}
        instance_path = env_config['instance_path']
        self.jobs = 0
        self.machines = 0
        self.step_per_operation = None
        self.instance_matrix = None
        # initial values for variables used for representation
        self.start_timestamp = datetime.datetime.now().timestamp()
        self.sum_op = 0
        self.max_time_op = float('-inf')
        instance_file = open(instance_path, 'r')
        # parsing is done in two times
        # first we get the metadata, e.g: the number of jobs, machine, op per job and machine per op
        line_str = instance_file.readline()
        line_cnt = 1
        while line_str:
            split_data = line_str.split()
            if line_cnt == 1:
                self.jobs, self.machines, self.machine_per_op = int(split_data[0]), int(split_data[1]), int(split_data[2])
                # contains the number of op for every job
                self.nb_op_job = np.zeros(self.jobs, dtype=int)
                # contains all the time to complete jobs
                self.min_jobs_length = np.zeros(self.jobs, dtype=int)
                self.max_jobs_length = np.zeros(self.jobs, dtype=int)
            else:
                i = 0
                # we get the actual jobs
                job_nb = line_cnt - 2
                self.nb_op_job[job_nb] = int(split_data[0])
            line_str = instance_file.readline()
            line_cnt += 1
        self.max_op_nb = max(self.nb_op_job)
        self.instance_matrix = np.full((self.jobs, self.max_op_nb, self.machines), fill_value=-1, dtype=int)
        self.compatible_machine_job = np.zeros((self.jobs, self.max_op_nb, self.machines), dtype=bool)
        instance_file.seek(0)
        line_str = instance_file.readline()
        line_cnt = 1
        while line_str:
            split_data = line_str.split()
            if line_cnt == 1:
                pass
            else:
                i = 1
                # we get the actual jobs
                job_nb = line_cnt - 2
                # current job's op
                op = 0
                while i < len(split_data):
                    # the number of machine for the op
                    machines = int(split_data[i])
                    i += 1
                    min_length = float('inf')
                    max_length = float('-inf')
                    for j in range(0, machines):
                        machine, time = int(split_data[i]), int(split_data[i + 1])
                        min_length = min(min_length, time)
                        max_length = max(min_length, time)
                        # /!\ WARNING in the encoding machine start at 1!!!!
                        self.instance_matrix[job_nb][op][machine - 1] = time
                        self.compatible_machine_job[job_nb][op][machine - 1] = True
                        self.max_time_op = max(self.max_time_op, time)
                        self.sum_op += time
                        i += 2
                    self.min_jobs_length[job_nb] += min_length
                    self.max_jobs_length[job_nb] += max_length
            line_str = instance_file.readline()
            line_cnt += 1
        instance_file.close()
        self.max_time_jobs = max(self.max_jobs_length)
        # check the parsed data are correct
        assert self.max_time_op > 0
        assert self.max_time_jobs > 0
        assert self.jobs > 0
        assert self.machines > 1, 'We need at least 2 machines'
        assert self.instance_matrix is not None
        self.max_time_jobs = max(self.max_jobs_length)
        # allocate a job + one to wait
        self.action_space = gym.spaces.Discrete(self.jobs + 1)
        # used for plotting
        self.colors = [
            tuple([random.random() for _ in range(3)]) for _ in range(self.machines)
        ]
        '''
        matrix with the following attributes for each job:
            -Legal job
            -Left over time on the current op
            -Current operation %
            -Total left over time
            -When next machine available
            -Time since IDLE: 0 if not available, time otherwise
            -Total IDLE time in the schedule
        '''
        self.observation_space = gym.spaces.Dict({
            "action_mask": gym.spaces.Box(0, 1, shape=(self.jobs + 1,)),
            "real_obs": gym.spaces.Box(low=0.0, high=1.0, shape=(self.jobs, 7), dtype=float),
        })