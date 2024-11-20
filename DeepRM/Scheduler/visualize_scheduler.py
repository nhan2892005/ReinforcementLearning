import pygame
import sys
from typing import List, Dict
from process import Process, ProcessInfo
from scheduler import Scheduler
import time

class SchedulerVisualizer:
    def __init__(self, width=1200, height=600):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Job Scheduler Visualization")
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.GRAY = (200, 200, 200)
        
        # Dimensions
        self.timeline_height = 100
        self.process_height = 40
        self.margin = 20
        
        # Fonts
        self.font = pygame.font.Font(None, 24)
        
        # Visualization state
        self.current_time = 0
        self.process_colors: Dict[str, tuple] = {}
        self.execution_history = []

        # Add stats tracking
        self.process_stats = {
            'turnaround_times': [],
            'waiting_times': [],
            'finished_count': 0,
            'total_time': 0
        }

        # Stats tracking
        self.stats = {
            'finished_processes': [],
            'current_time': 0
        }

    def _draw_stats_box(self):
        """Draw statistics box in bottom right"""
        box_width = 400
        box_height = 180
        margin = 20
        x = self.width - box_width - margin
        y = self.height - box_height - margin
        
        # Draw box background
        pygame.draw.rect(self.screen, self.GRAY, (x, y, box_width, box_height))
        pygame.draw.rect(self.screen, self.BLACK, (x, y, box_width, box_height), 2)
        
        # Calculate stats
        if self.stats['finished_processes']:
            total_processes = len(self.stats['finished_processes'])
            avg_turnaround = sum(p.turnaround_time for p in self.stats['finished_processes']) / total_processes
            avg_waiting = sum(p.waiting_time for p in self.stats['finished_processes']) / total_processes
            throughput = total_processes / self.stats['current_time'] if self.stats['current_time'] > 0 else 0
        else:
            avg_turnaround = avg_waiting = throughput = 0
        
        # Draw stats text
        texts = [
            "Statistics",
            f"Average turnaround time:        {avg_turnaround:.1f} seconds",
            f"Average waiting time:           {avg_waiting:.1f} seconds",
            f"Throughput:                     {throughput:.2f} processes per second",
            f"Completed processes:            {len(self.stats['finished_processes'])}"
        ]

        for i, text in enumerate(texts):
            surface = self.font.render(text, True, self.BLACK)
            self.screen.blit(surface, (x + 10, y + 10 + i * 30))
    
    def _assign_process_color(self, pid: str):
        if pid not in self.process_colors:
            # Generate unique color for process
            color = (
                (hash(pid) & 0xFF0000) >> 16,
                (hash(pid) & 0x00FF00) >> 8,
                hash(pid) & 0x0000FF
            )
            self.process_colors[pid] = color

    def _draw_timeline(self, processes: List[Process], current_process: Process):
        # Draw timeline
        pygame.draw.rect(self.screen, self.WHITE, 
                        (self.margin, self.margin, 
                         self.width - 2*self.margin, self.timeline_height))
        
        # Draw time markers
        for i in range(0, self.width - 2*self.margin, 50):
            time = i // 5
            text = self.font.render(str(time), True, self.BLACK)
            self.screen.blit(text, (i + self.margin, self.timeline_height + self.margin))
            
        # Draw execution history
        for entry in self.execution_history:
            start_time, end_time, pid = entry
            if pid in self.process_colors:
                x = self.margin + start_time * 5
                width = (end_time - start_time) * 5
                pygame.draw.rect(self.screen, self.process_colors[pid],
                               (x, self.margin, width, self.timeline_height))

    def _draw_ready_queue(self, processes: List[Process], y_start: int):
        # Draw ready queue
        text = self.font.render("Ready Queue:", True, self.WHITE)
        self.screen.blit(text, (self.margin, y_start))
        
        for i, process in enumerate(processes):
            self._assign_process_color(process.pid)
            y = y_start + (i+1) * (self.process_height + 5)
            pygame.draw.rect(self.screen, self.process_colors[process.pid],
                           (self.margin, y, 200, self.process_height))
            
            text = self.font.render(f"PID: {process.pid}", True, self.BLACK)
            self.screen.blit(text, (self.margin + 10, y + 10))

    def _draw_current_process(self, process: Process, y_start: int):
        if process:
            text = self.font.render("Currently Executing:", True, self.WHITE)
            self.screen.blit(text, (self.width - 300, y_start))
            
            self._assign_process_color(process.pid)
            pygame.draw.rect(self.screen, self.process_colors[process.pid],
                           (self.width - 300, y_start + 40, 250, self.process_height))
            
            text = self.font.render(f"PID: {process.pid}", True, self.BLACK)
            self.screen.blit(text, (self.width - 290, y_start + 50))

    def update(self, scheduler: Scheduler, delta_time: float):
        self.screen.fill(self.BLACK)
        
        # Update execution history for current process
        if scheduler._executing_process:
            pid = scheduler._executing_process.pid
            self.execution_history.append((
                self.current_time,
                self.current_time + delta_time,
                pid
            ))
            self._assign_process_color(pid)

        # Update stats
        self.stats['current_time'] += delta_time
        if scheduler._executing_process and scheduler._executing_process.is_finished:
            if scheduler._executing_process not in self.stats['finished_processes']:
                self.stats['finished_processes'].append(scheduler._executing_process)
        
        # Draw components
        self._draw_timeline(list(scheduler._ready_queue), scheduler._executing_process)
        self._draw_ready_queue(list(scheduler._ready_queue), self.timeline_height + 60)
        self._draw_current_process(scheduler._executing_process, self.timeline_height + 60)
        self._draw_stats_box()

        self.current_time += delta_time
        pygame.display.flip()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

def visualize_scheduler(scheduler: Scheduler, processes: List[Process], speed=1.0):
    vis = SchedulerVisualizer()
    clock = pygame.time.Clock()
    
    # Bind visualization update to process events
    def on_process_executed(process: Process, allocated_time: int, execution_time: int):
        vis.update(scheduler, execution_time)
        time.sleep(execution_time * speed)
        clock.tick(60)
        vis.handle_events()
    
    for process in processes:
        process.register_on_executed_listener(on_process_executed)
    
    # Run simulation
    for process in processes:
        time_till_arrival = max(0, process.process_info.arrival_time - scheduler.time_elapsed)
        scheduler.increase_time(time_till_arrival)
        scheduler.enqueue_process(process)
        vis.update(scheduler, time_till_arrival)
        time.sleep(time_till_arrival * speed)
        clock.tick(60)
        vis.handle_events()
    
    scheduler.finish()
    
    # Keep window open until closed
    while True:
        vis.handle_events()
        clock.tick(60)

def read_jobs_from_file(filename: str) -> List[ProcessInfo]:
    """Read jobs from file and create ProcessInfo objects"""
    processes = []
    with open(filename, 'r') as f:
        for line in f:
            # Skip empty lines
            if not line.strip():
                continue
            # Parse line: PID arrival_time burst_time
            pid, arrival_time, burst_time = line.split()
            processes.append(ProcessInfo(
                pid,
                int(arrival_time),
                int(burst_time)
            ))
    return processes

# Usage example
if __name__ == "__main__":
    from simulate import create_simulation_processes
    from fcfs import FirstComeFirstServedScheduler
    from srt  import ShortestRemainingTimeScheduler
    from round_robin import RoundRobinScheduler
    
    process_info = read_jobs_from_file('DeepRM/Scheduler/job_data.txt')
    processes = create_simulation_processes(process_info)
    
    scheduler = ShortestRemainingTimeScheduler()
    visualize_scheduler(scheduler, processes, speed=0.5)