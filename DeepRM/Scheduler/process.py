from collections import deque


class ProcessInfo:
    """
    Contains data required for simulating how a process is executed in a scheduler.
    """
    def __init__(self, pid: str, arrival_time: int, processing_time: int):
        self.pid = pid
        """The PID to be used by the simulation"""
        self.arrival_time = arrival_time
        """The time that the process will arrive and be enqueued into the scheduler"""
        self.processing_time = processing_time
        """The amount of execution time required before the process finishes"""


class Process:
    """
    A mock process used for simulating process information.
    """
    def __init__(self, process_info: ProcessInfo):
        self.process_info = process_info
        """The simulation information associated with this process"""
        self.__time_processed = 0
        self.__waiting_time = 0
        self.__on_finished_listeners = deque()
        self.__on_executing_listeners = deque()
        self.__on_executed_listeners = deque()

    @property
    def pid(self):
        """
            :return:
                The process id for the process in question.
        """
        return self.process_info.pid

    @property
    def is_finished(self):
        """
            :return:
                Whether or not the process no longer needs to be executed.
        """
        return self.process_info.processing_time <= self.time_processed

    def allocate_time(self, allocated_time: int):
        """
        Executes the process for a given amount of time.
            :param allocated_time:
                The time given to execute the process.
            :return:
                The time left over from the execution.
        """
        self.__on_executing(allocated_time)
        excess_time = 0
        if self.remaining_time < allocated_time:
            # If we have more time to execute than is required then we need to account for that
            excess_time = allocated_time - self.remaining_time
            self.__time_processed = self.total_time_to_process
        else:
            self.__time_processed += allocated_time
        self.__on_executed(allocated_time, allocated_time - excess_time)
        if self.is_finished:
            self.__on_finished()
        return excess_time

    def wait(self, waiting_time: int):
        """
        Makes the process wait i.e. time passes but the process does not do anything.
            :param waiting_time:
                The amount of time the process waits for.
        """
        self.__waiting_time += waiting_time

    @property
    def time_processed(self):
        """
            :return:
                The amount of time that the process has been processed for.
        """
        return self.__time_processed

    @property
    def waiting_time(self):
        """
            :return:
                The amount of time that the process has waited for without being executed.
        """
        return self.__waiting_time

    @property
    def total_time_to_process(self):
        """
            :return:
                The total amount of time that it takes for the process to be finished.
        """
        return self.process_info.processing_time

    @property
    def turnaround_time(self):
        """
            :return:
                The time that it took for the process to finish since the processed was enqueued by a scheduler.
        """
        assert self.time_processed >= self.process_info.processing_time, \
            "The process {} has not finished: {} time processed vs {} total time needed".format(
                self.process_info.pid,
                self.time_processed,
                self.total_time_to_process
            )
        return self.time_processed + self.waiting_time

    @property
    def remaining_time(self):
        """
            :return:
                The estimated amount of time remaining before the process finishes executing.
        """
        return self.process_info.processing_time - self.time_processed

    def register_on_executed_listener(self, listener):
        """
        Adds a listener in the form of a lamda function which is called whenever the process has finished
        executing. The lambda function takes in the process, the time allocated to the process for execution and the
        time that the process actually executed for.
            :param listener:
                A lamda function listener.
        """
        self.__on_executed_listeners.append(listener)

    def register_on_executing_listener(self, listener):
        """
        Adds a listener in the form of a lambda function which is called whenever the process
        begins to execute. The lambda function takes in the process as a parameter as well as
        how long the process will execute for. This does not account for the process finishing early.
            :param listener:
                A lambda function listener.
        """
        self.__on_executing_listeners.append(listener)

    def register_on_finished_listener(self, listener):
        """
        Adds a listener in the form of a lambda function which is called when the process is finished. The lambda
        function takes in the process as a parameter.
        """
        self.__on_finished_listeners.append(listener)

    def __on_executed(self, allocated_time: int, time_executed: int):
        """
        Calls all on executed listeners. This happens after the process has executed for the allocated amount of time.
            :param allocated_time:
                The time that was allocated to the process for execution.
            :param time_executed:
                The amount of time that the process actually executed for.
        """
        for listener in self.__on_executed_listeners:
            listener(self, allocated_time, time_executed)

    def __on_executing(self, allocated_time: int):
        """
        Calls all on executing listeners.
        """
        for listener in self.__on_executing_listeners:
            listener(self, allocated_time)

    def __on_finished(self):
        """
        Calls all on finished listeners.
        """
        for listener in self.__on_finished_listeners:
            listener(self)

    def __str__(self):
        return self.process_info.pid