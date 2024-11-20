from fcfs import FirstComeFirstServedScheduler

from process import Process


class ShortestRemainingTimeScheduler(FirstComeFirstServedScheduler):
    """
    A preemptive scheduler that executes whichever process is estimated to finish first.
    Note that the FCFS scheduler's increase_time(...) method is utilised for this scheduler since the preemption and
    shortest remaining times are calculated during the enqueuing of the process.
    """
    def enqueue_process(self, process: Process):
        """
        Enqueues a process for execution such that the queue remains ordered by the remaining time, starting from
        the shortest remaining time to the longest remaining time.
            :param process:
                The process to be enqueued by the scheduler
        """
        self._on_process_enqueuing(process)
        if self._executing_process is not None:
            if self._executing_process.remaining_time > process.remaining_time:
                """
                If the currently executing process is estimated to finish after the newly enqueued process, then
                preempt the process and push it back onto the start of the ready queue.
                Allow the new process to start executing (since it is estimated to finish first)
                """
                self._ready_queue.appendleft(self._executing_process)
                self._executing_process = process
                return
        # Figure out where abouts this process belongs within the queue
        for p in range(len(self._ready_queue)-1, -1, -1):
            other_process = self._ready_queue[p]
            if other_process.remaining_time <= process.remaining_time:
                self._ready_queue.insert(p + 1, process)
                return
        """
        At this point the process must belong at the start of the queue since we already checked if it was meant to be
        executing and we checked everywhere else along the queue.
        """
        self._ready_queue.appendleft(process)