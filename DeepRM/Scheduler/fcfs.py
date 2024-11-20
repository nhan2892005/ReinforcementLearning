from scheduler import Scheduler


class FirstComeFirstServedScheduler(Scheduler):
    """
    A simple non-preemptive scheduler that executes processes in a FIFO manner.
    I.e. The nth enqueued process will be the nth process to finish.
    """
    def increase_time(self, time: int):
        super(FirstComeFirstServedScheduler, self).increase_time(time)
        while time > 0:
            if self._executing_process is None and len(self._ready_queue) > 0:
                # Select the process in the queue to be executed if we aren't executing one already.
                self._executing_process = self._ready_queue.popleft()
            if self._executing_process is not None:
                # Allocate the time to the process
                excess_time = self._executing_process.allocate_time(time)
                # Calculate how much time the process actually used to execute
                time_executed = time - excess_time
                # Make all the processes in the ready queue wait for however much time was used by the process
                for process in self._ready_queue:
                    process.wait(time_executed)
                if self._executing_process.is_finished:
                    # If the process finished, remove the executing process so that a new process may start executing
                    self._executing_process = None
            else:
                # At this point there mustn't be any processes in the ready queue
                time_executed = time
            time -= time_executed
        assert time == 0, "The amount of time left was not 0. It was actually, {}.".format(time)