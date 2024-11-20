from scheduler import Scheduler


class RoundRobinScheduler(Scheduler):
    """
    A preemptive scheduler that provides each process a time slice to execute in. The length of the allocated time
    slice is specified by a time quantum which, by default is 2.
    """
    def __init__(self, time_quantum=2):
        """
            :param time_quantum:
                The amount of time that will be allocated to each process for a given time slice.
        """
        super().__init__()
        self.__time_quantum = time_quantum
        """The amount of time given to each time slice"""
        self.__time_left_for_time_slice = self.__time_quantum
        """The amount of time that is left for the time slice of the currently executing process"""

    @property
    def time_quantum(self) -> int:
        """
            :return:
                The amount of time that will be allocated to each process for a given time slice.
        """
        return self.__time_quantum

    def reset_current_time_slice(self):
        """
        Resets the remaining time left on the time slice, back to the time quantum. Note that this should only be
        called once a process has finished or the time slice for the currently executed time slice is up.
        """
        self.__time_left_for_time_slice = self.__time_quantum

    def increase_time(self, time: int):
        super(RoundRobinScheduler, self).increase_time(time)
        while time > 0:
            if self._executing_process is not None and self.__time_left_for_time_slice <= 0:
                # If the currently executing process's time slice has expired then stop processing the process
                self._ready_queue.append(self._executing_process)
                self._executing_process = None
                self.reset_current_time_slice()
            if self._executing_process is None and len(self._ready_queue) > 0:
                # If we don't have any processes
                self._executing_process = self._ready_queue.popleft()
            if self._executing_process is not None:
                # Figure out how much time we can allocate to the process
                allocated_time = min(self.__time_left_for_time_slice, time)
                # Execute the process and figure out how much time the process had in excess
                excess_allocated_time = self._executing_process.allocate_time(allocated_time)
                # Figure out how much time the process actually executed
                time_executed = allocated_time - excess_allocated_time
                self.__time_left_for_time_slice -= time_executed
                # Make the other processes wait for the same amount of time
                for process in self._ready_queue:
                    process.wait(time_executed)
                if self._executing_process.is_finished:
                    # If the process finished we need to remove it so that other processes may start executing
                    self._executing_process = None
                    self.reset_current_time_slice()
            else:
                # Dispatcher is idle for 'time' seconds since there are no processes to process
                time_executed = time
            time -= time_executed
        assert time == 0, "The amount of time elapsed was not 0. It was actually, {}.".format(time)