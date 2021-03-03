import random
from pathlib import Path

import gym
import datetime
import bisect
import pandas as pd
import numpy as np
import plotly.figure_factory as ff

class FlexibleJssEnv(gym.Env):

    def __init__(self, env_config=None):
        if env_config is None:
            env_config = {'instance_path': str(Path(__file__).parent.absolute()) + '/instances/flexible/mt10c1.fjs'}
        instance_path = env_config['instance_path']
        self.jobs = 0
        self.machines = 0
        self.step_per_operation = None
        self.instance_matrix = None
        self.state = None
        self.legal_actions = None
        self.current_time_step = 0
        self.next_time_step = list()
        # representation data
        self.total_idle_time_jobs = None
        self.idle_time_jobs_last_op = None
        self.total_perform_op_time_jobs = None
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
                    op += 1
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
            "action_mask": gym.spaces.Box(0, 1, shape=(self.jobs,)),
            "real_obs": gym.spaces.Box(low=0.0, high=1.0, shape=(self.jobs, 7), dtype=float),
        })

    def get_legal_actions(self):
        return self.legal_actions

    def _get_current_state_representation(self):
        self.state[:, 0] = self.legal_actions
        return {
            "real_obs": self.state,
            "action_mask": self.legal_actions,
        }

    def reset(self):
        self.current_time_step = 0
        self.next_time_step = list()
        self.nb_legal_actions = self.jobs
        # represent all the legal actions
        self.legal_actions = np.ones(self.jobs, dtype=bool)
        # used to represent the solution
        self.solution = np.full((self.jobs, self.max_op_nb, self.machines), -1, dtype=int)
        self.time_until_available_machine = np.zeros(self.machines, dtype=int)
        self.time_until_finish_current_op_jobs = np.zeros(self.jobs, dtype=int)
        self.todo_time_step_job = np.zeros(self.jobs, dtype=int)
        self.machine_legal = np.zeros(self.machines, dtype=bool)
        # state rep
        self.total_idle_time_jobs = np.zeros(self.jobs, dtype=int)
        self.idle_time_jobs_last_op = np.zeros(self.jobs, dtype=int)
        self.total_perform_op_time_jobs = np.zeros(self.jobs, dtype=int)
        self.state = np.zeros((self.jobs, 7), dtype=float)
        return self._get_current_state_representation()

    def get_machine_needed_job(self, job_id):
        time_step_to_do = self.todo_time_step_job[job_id]
        return np.where(self.compatible_machine_job[job_id][time_step_to_do])[0][0]

    def step(self, action: int):
        reward = 0.0
        current_time_step_job = self.todo_time_step_job[action]
        machine_needed = self.get_machine_needed_job(action)
        time_needed = self.instance_matrix[action][current_time_step_job][machine_needed]
        reward += time_needed
        self.time_until_available_machine[machine_needed] = time_needed
        self.time_until_finish_current_op_jobs[action] = time_needed
        to_add_time_step = self.current_time_step + time_needed
        if to_add_time_step not in self.next_time_step:
            index = bisect.bisect_left(self.next_time_step, to_add_time_step)
            self.next_time_step.insert(index, to_add_time_step)
        self.solution[action][current_time_step_job][machine_needed] = self.current_time_step
        for job in range(self.jobs):
            if self.todo_time_step_job[job] < self.nb_op_job[job] and self.get_machine_needed_job(job) == machine_needed and self.legal_actions[job]:
                self.legal_actions[job] = False
                self.nb_legal_actions -= 1
        self.machine_legal[machine_needed] = False
        while self.nb_legal_actions == 0 and len(self.next_time_step) > 0:
            reward -= self._increase_time_step()
        return self._get_current_state_representation(), reward, self._is_done(), {}

    def _is_done(self):
        if self.nb_legal_actions == 0:
            self.last_time_step = self.current_time_step
            return True
        return False

    def _increase_time_step(self):
        """
        The heart of the logic his here, we need to increase every counter when we have a nope action called
        and return the time elapsed
        :return: time elapsed
        """
        hole_planning = 0
        next_time_step_to_pick = self.next_time_step.pop(0)
        difference = next_time_step_to_pick - self.current_time_step
        self.current_time_step = next_time_step_to_pick
        for job in range(self.jobs):
            was_left_time = self.time_until_finish_current_op_jobs[job]
            if was_left_time > 0:
                performed_op_job = min(difference, was_left_time)
                self.time_until_finish_current_op_jobs[job] = max(0, self.time_until_finish_current_op_jobs[
                    job] - difference)
                self.total_perform_op_time_jobs[job] += performed_op_job
                if self.time_until_finish_current_op_jobs[job] == 0:
                    self.todo_time_step_job[job] += 1
                    self.total_idle_time_jobs[job] += (difference - was_left_time)
                    self.idle_time_jobs_last_op[job] = (difference - was_left_time)
            elif self.todo_time_step_job[job] < self.nb_op_job[job]:
                self.total_idle_time_jobs[job] += difference
                self.idle_time_jobs_last_op[job] += difference
        for machine in range(self.machines):
            if self.time_until_available_machine[machine] < difference:
                empty = difference - self.time_until_available_machine[machine]
                hole_planning += empty
            self.time_until_available_machine[machine] = max(0, self.time_until_available_machine[
                machine] - difference)
            if self.time_until_available_machine[machine] == 0:
                for job in range(self.jobs):
                    if self.todo_time_step_job[job] < self.nb_op_job[job] and self.get_machine_needed_job(job) == machine and not self.legal_actions[job]:
                        self.legal_actions[job] = True
                        self.nb_legal_actions += 1
                        if not self.machine_legal[machine]:
                            self.machine_legal[machine] = True
        return hole_planning

    def render(self, mode='human'):
        df = []
        for job in range(self.jobs):
            i = 0
            while i < self.nb_op_job[job]:
                for machine in range(self.machines):
                    if self.solution[job][i][machine] != -1:
                        dict_op = dict()
                        dict_op["Task"] = 'Job {}'.format(job)
                        start_sec = self.start_timestamp + self.solution[job][i][machine]
                        finish_sec = start_sec + self.instance_matrix[job][i][machine]
                        dict_op["Start"] = datetime.datetime.fromtimestamp(start_sec)
                        dict_op["Finish"] = datetime.datetime.fromtimestamp(finish_sec)
                        dict_op["Resource"] = "Machine {}".format(machine)
                        df.append(dict_op)
                i += 1
        fig = None
        if len(df) > 0:
            df = pd.DataFrame(df)
            fig = ff.create_gantt(df, index_col='Resource', colors=self.colors, show_colorbar=True,
                                  group_tasks=True)
            fig.update_yaxes(autorange="reversed")  # otherwise tasks are listed from the bottom up
        return fig