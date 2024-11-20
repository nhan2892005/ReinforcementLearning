from abc import abstractmethod
from collections import deque

from process import Process


class Scheduler:
    """
    Handles when, which and for how long a process is executed by a simulated processor.
    """

    def __init__(self):
        self._executing_process = None
        """The process that is currently being executed by the scheduler"""
        self._ready_queue = deque()
        """The queue of processes that are waiting their turn to be executed"""
        self._time_elapsed = 0
        """The amount of time that has elapsed in the scheduler simulation"""
        self.__on_process_enqueuing_listeners = deque()
        """
        A deque of listeners. Mainly used for debugging purposes.
        E.g. Printing out a process when it is enqueued
        """

    def enqueue_process(self, process: Process):
        """
        Adds a process that needs to be executed
            :param process:
                The process being added to the queue
        """
        self._on_process_enqueuing(process)
        self._ready_queue.append(process)

    def _on_process_enqueuing(self, process: Process):
        """
        Notifies all process enqueuing listeners
            :param process:
                The process being enqueued
        """
        for listener in self.__on_process_enqueuing_listeners:
            listener(process)

    def register_on_process_enqueuing_listener(self, listener):
        """
        Registers a listener lambda function to the scheduler which is called whenever a process is enqueued.
        The lambda listener must take a the process that is being enqueued as a parameter.
            :param listener:
                The listener being registered by the scheduler
        """
        self.__on_process_enqueuing_listeners.append(listener)

    @abstractmethod
    def increase_time(self, time: int):
        """
        Increases the time that has elapsed in the scheduler's simulation. Note that the scheduler will be idle when
        there are no processes in the ready queue.
            :param time:
                The amount of time that the elapsed time is increase by
        """
        self._time_elapsed += time

    @property
    def time_to_finish_processes(self) -> int:
        """
        :return:

        """
        time_to_finish = 0
        for process in self._ready_queue:
            time_to_finish += process.remaining_time
        if self._executing_process is not None:
            time_to_finish += self._executing_process.remaining_time
        return time_to_finish

    @property
    def time_elapsed(self):
        """
            :return:
                The amount of time that has elapsed since the beginning of the scheduler simulation
        """
        return self._time_elapsed

    def finish(self):
        """
            Increases the time such that all processes finish executing.
        """
        time_to_execute = self.time_to_finish_processes
        self.increase_time(time_to_execute)
        assert self._executing_process is None, "There is apparently a process {} that is still currently executing. " \
                                                "Required time: {}".format(
            self._executing_process,
            self._executing_process.remaining_time
        )