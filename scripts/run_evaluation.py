import argparse
import asyncio
import json
import logging
import os
import sys
import time
import queue
from datetime import datetime
from dotenv import load_dotenv
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.text import Text
from rich.layout import Layout
from rich.logging import RichHandler
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from moatless.benchmark.evaluation_v2 import (
    EvaluationStatus, InstanceStatus, create_evaluation_name,
    TreeSearchSettings, EvaluationRunner
)
from moatless.benchmark.repository import EvaluationFileRepository
from moatless.completion.completion import CompletionModel, LLMResponseFormat
from moatless.schema import MessageHistoryType
from moatless.agent.settings import AgentSettings
from moatless.benchmark.evaluation_factory import create_evaluation

# Configure root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

logging.getLogger("LiteLLM").setLevel(logging.WARNING)

load_dotenv()

class LogPanel:
    def __init__(self, max_lines=100, visible_lines=20):
        self.logs = deque(maxlen=max_lines)
        self.visible_lines = visible_lines
        
    def write(self, message):
        self.logs.append(message.strip())
    
    def get_panel(self):
        # Get the most recent logs up to visible_lines in reverse order
        visible_logs = list(self.logs)[-self.visible_lines:]
        visible_logs.reverse()  # Reverse to show newest first
        log_text = Text("\n".join(visible_logs))
        return Panel(
            log_text,
            title=f"Logs (newest first, showing {len(visible_logs)} of {len(self.logs)} messages)",
            border_style="blue",
            height=self.visible_lines + 2  # +2 for panel borders
        )

class UILogger(logging.Handler):
    def __init__(self, log_panel):
        super().__init__()
        self.log_panel = log_panel
        self.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    def emit(self, record):
        try:
            msg = self.format(record)
            self.log_panel.write(msg)
        except Exception:
            self.handleError(record)

class EvaluationMonitor:
    def __init__(self, repository, evaluation):
        self.repository = repository
        self.evaluation = evaluation
        self.console = Console()
        self.start_time = time.time()
        self.instances_data = {}
        self.live = None
        self.log_panel = LogPanel(max_lines=1000, visible_lines=25)
        self.logger = logger
        self.event_queue = queue.Queue()
        self.needs_update = False
        self.needs_stats_update = False
        self.last_log_update = 0
        
        # Remove existing handlers and add UI handler
        self.logger.handlers = []
        self.logger.addHandler(UILogger(self.log_panel))
        
        # Load initial instances
        for instance in self.repository.list_instances(self.evaluation.evaluation_name):
            self.instances_data[instance.instance_id] = {
                'status': instance.status,
                'duration': instance.duration,
                'resolved': instance.resolved,
                'error': instance.error
            }
        
        self.logger.info(f"Starting evaluation monitor for {evaluation.evaluation_name}")
        self.logger.info(f"Found {len(self.instances_data)} instances to evaluate")

    def handle_event(self, event):
        """Handle evaluation events by putting them in the queue"""
        try:
            self.event_queue.put_nowait(event)
            self.needs_update = True  # Mark that we need an update
        except queue.Full:
            self.logger.warning("Event queue is full, dropping event")

    async def process_event(self, event):
        """Process a single event"""
        event_type = event.event_type
        data = event.data if event.data else {}
        
        instance_id = data.get("instance_id")
        
        if event_type == "evaluation_started":
            self.logger.info("Evaluation started")
            return

        if not instance_id and event_type != "evaluation_started":
            self.logger.warning(f"Instance ID not found in event data: {data}")
            return

        if event_type == "instance_started":
            self.instances_data.setdefault(instance_id, {}).update({
                'status': InstanceStatus.STARTED,
                'duration': None,
                'resolved': None,
                'error': None,
                'start_time': time.time()
            })
            self.logger.info(f"Started instance: {instance_id}")
            
        elif event_type == "instance_completed":
            self.instances_data.setdefault(instance_id, {}).update({
                'status': InstanceStatus.COMPLETED,
                'resolved': data.get("resolved", False),
                'duration': data.get("duration"),
                'error': None,
                'iteration': data.get("iteration", 0),
                'total_cost': data.get("total_cost", 0),
                'best_reward': data.get("best_reward", 0),
                'prompt_tokens': data.get("prompt_tokens", 0),
                'completion_tokens': data.get("completion_tokens", 0),
                'cached_tokens': data.get("cached_tokens", 0)
            })
            self.needs_stats_update = True
            self.logger.info(f"Completed instance: {instance_id} (resolved: {data.get('resolved', False)})")
            
        elif event_type == "instance_error":
            self.instances_data.setdefault(instance_id, {}).update({
                'status': InstanceStatus.ERROR,
                'resolved': False,
                'duration': None,
                'error': data.get("error"),
                'iteration': data.get("iteration", 0),
                'total_cost': data.get("total_cost", 0),
                'best_reward': data.get("best_reward", 0),
                'prompt_tokens': data.get("prompt_tokens", 0),
                'completion_tokens': data.get("completion_tokens", 0),
                'cached_tokens': data.get("cached_tokens", 0)
            })
            self.needs_stats_update = True
            self.logger.error(f"Error in instance {instance_id}: {data.get('error')}")
            
        elif event_type == "tree_progress":
            if instance_id in self.instances_data:
                self.instances_data[instance_id].update({
                    'iteration': data["iteration"],
                    'total_cost': data["total_cost"],
                    'best_reward': data["best_reward"]
                })
                self.logger.info(f"Progress for {instance_id}: iter={data['iteration']}, cost={data['total_cost']:.2f}, reward={data['best_reward']:.2f}")

    async def process_events(self):
        """Process events from the queue"""
        while True:
            try:
                # Check queue in a non-blocking way
                while not self.event_queue.empty():
                    event = self.event_queue.get_nowait()
                    await self.process_event(event)
                    self.event_queue.task_done()
                
                current_time = time.time()
                
                # Update logs once per second
                if current_time - self.last_log_update >= 1.0:
                    self.needs_update = True
                    self.last_log_update = current_time
                
                # Update display if needed
                if self.live and (self.needs_update or self.needs_stats_update):
                    self.live.update(self._create_layout())
                    self.needs_update = False
                    self.needs_stats_update = False
                
                await asyncio.sleep(1.0)
            except queue.Empty:
                pass  # Queue is empty, continue
            except Exception as e:
                self.logger.error(f"Error processing event: {e}")
                await asyncio.sleep(1.0)

    def create_progress_table(self):
        """Create a rich table showing evaluation progress"""
        table = Table(title=f"Evaluation Progress: {self.evaluation.evaluation_name}")
        
        table.add_column("Instance ID", style="cyan")
        table.add_column("Status", style="magenta")
        table.add_column("Duration", style="green")
        table.add_column("Resolved", style="yellow")
        table.add_column("Iterations", style="blue")
        table.add_column("Tokens", style="blue")
        table.add_column("Progress", style="blue")
        
        # Sort instances: running first (by start time desc), then others
        sorted_instances = []
        running_instances = []
        other_instances = []
        
        for instance_id, data in self.instances_data.items():
            if data.get('status') == InstanceStatus.STARTED:
                running_instances.append((instance_id, data))
            else:
                other_instances.append((instance_id, data))
        
        # Sort running instances by start time (most recent first)
        running_instances.sort(key=lambda x: x[1].get('start_time', 0), reverse=True)
        sorted_instances = running_instances + other_instances
        
        for instance_id, data in sorted_instances:
            status = data.get('status', 'pending')
            duration = f"{data.get('duration', 0):.1f}s" if data.get('duration') else "-"
            resolved = "✓" if data.get('resolved') else "✗" if data.get('resolved') is False else "-"
            
            # Get iterations and tokens
            iterations = data.get('iteration', 0)
            tokens = int(data.get('total_cost', 0) * 1000) if data.get('total_cost') is not None else 0
            
            # Add progress info if available
            progress = ""
            if 'iteration' in data:
                progress = f"Reward: {data['best_reward']:.2f}"
            
            status_style = {
                'pending': 'white',
                'started': 'yellow',
                'completed': 'green',
                'error': 'red'
            }.get(status, 'white')
            
            table.add_row(
                instance_id,
                Text(status, style=status_style),
                duration,
                resolved,
                str(iterations),
                f"{tokens:,}",
                progress
            )
        
        return table

    def create_stats_panel(self):
        """Create a panel showing evaluation statistics"""
        total = len(self.instances_data)
        completed = sum(1 for i in self.instances_data.values() if i['status'] == InstanceStatus.COMPLETED)
        errors = sum(1 for i in self.instances_data.values() if i['status'] == InstanceStatus.ERROR)
        running = sum(1 for i in self.instances_data.values() if i['status'] == InstanceStatus.STARTED)
        resolved = sum(1 for i in self.instances_data.values() if i.get('resolved', False))
        
        text = Text()
        text.append(f"Total Instances: {total}\n", style="cyan")
        text.append(f"Completed: {completed}\n", style="green")
        text.append(f"Running: {running}\n", style="yellow")
        text.append(f"Errors: {errors}\n", style="red")
        text.append(f"Success Rate: {(resolved/total*100 if total > 0 else 0):.1f}%\n", style="magenta")
        text.append(f"Elapsed Time: {self._format_elapsed_time()}\n", style="blue")
        
        return Panel(text, title="Evaluation Statistics", border_style="bright_blue")

    def create_info_panel(self):
        """Create a panel showing evaluation configuration"""
        evaluation = self.evaluation

        text = Text()
        # Model info
        text.append("Model Settings:\n", style="bold magenta")
        text.append(f"  Model: ", style="cyan")
        text.append(f"{evaluation.settings.model.model}\n", style="white")
        text.append(f"  Temperature: ", style="cyan")
        text.append(f"{evaluation.settings.model.temperature}\n", style="white")
        text.append(f"  Response Format: ", style="cyan")
        text.append(f"{evaluation.settings.model.response_format.value}\n", style="white")
        
        # Tree search settings
        text.append("\nTree Search Settings:\n", style="bold magenta")
        text.append(f"  Max Iterations: ", style="cyan")
        text.append(f"{evaluation.settings.max_iterations}\n", style="white")
        text.append(f"  Max Expansions: ", style="cyan")
        text.append(f"{evaluation.settings.max_expansions}\n", style="white")
        text.append(f"  Max Cost: ", style="cyan")
        text.append(f"{evaluation.settings.max_cost}\n", style="white")
        
        # Agent settings
        text.append("\nAgent Settings:\n", style="bold magenta")
        text.append(f"  Message History: ", style="cyan")
        text.append(f"{evaluation.settings.agent_settings.message_history_type.value}\n", style="white")
        
        return Panel(text, title="Evaluation Info", border_style="green")

    def _format_elapsed_time(self):
        """Format elapsed time in a human-readable format"""
        seconds = int(time.time() - self.start_time)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _create_layout(self):
        """Create the layout for the display"""
        layout = Layout()
        
        # Split into main content and right side
        layout.split_row(
            Layout(name="main", ratio=2),
            Layout(name="right", ratio=1)
        )
        
        # Split main content into progress and stats
        layout["main"].split_column(
            Layout(self.create_progress_table(), name="progress"),
            Layout(self.create_stats_panel(), name="stats", size=8)
        )
        
        # Split right side into info and logs
        layout["right"].split_column(
            Layout(self.create_info_panel(), name="info", size=15),
            Layout(self.log_panel.get_panel(), name="logs")
        )
        
        return layout

    async def start_monitoring(self):
        """Start monitoring the evaluation"""
        with Live(
            self._create_layout(),
            console=self.console,
            refresh_per_second=1,
            auto_refresh=True
        ) as self.live:
            # Start event processing task
            event_task = asyncio.create_task(self.process_events())
            
            while True:
                try:
                    evaluation = self.repository.load_evaluation(self.evaluation.evaluation_name)
                    if not evaluation:
                        self.logger.error("Evaluation not found!")
                        break
                    
                    if evaluation.status in [EvaluationStatus.COMPLETED, EvaluationStatus.ERROR]:
                        # Force final update of stats
                        self.needs_stats_update = True
                        self.live.update(self._create_layout())
                        break
                        
                    await asyncio.sleep(1.0)
                    
                except Exception as e:
                    self.logger.error(f"Error monitoring evaluation: {e}")
                    break
            
            # Cancel event processing task
            event_task.cancel()
            try:
                await event_task
            except asyncio.CancelledError:
                pass

def validate_evaluation_setup(repository, evaluation, args):
    """Validate evaluation setup and throw exceptions for any issues"""
    
    # Validate model configuration
    if not args.model:
        raise ValueError("Model name must be specified")
    
    # Check if evaluation exists
    if not evaluation:
        raise RuntimeError("Failed to create evaluation")

    return evaluation

def main():
    parser = argparse.ArgumentParser(description="Run a model evaluation with progress monitoring")
    parser.add_argument("--model", required=True, help="Model name (e.g., gemini/gemini-2.0-flash-exp)")
    parser.add_argument("--api-key", help="API key for the model")
    parser.add_argument("--base-url", help="Base URL for the API")
    parser.add_argument("--max-iterations", type=int, default=20, help="Maximum iterations per instance")
    parser.add_argument("--max-expansions", type=int, default=1, help="Maximum expansions per state")
    parser.add_argument("--max-cost", type=float, default=1.0, help="Maximum cost in tokens")
    parser.add_argument("--instance-ids", nargs="+", help="Specific instance IDs to evaluate")
    parser.add_argument("--min-resolved", type=int, default=20, help="Minimum number of resolved instances")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--split", default="lite", choices=["lite", "verified", "combo", "random"], 
                       help="Dataset split to use")
    parser.add_argument("--repos", nargs="+", help="Filter by specific repositories")
    parser.add_argument("--ignore-repos", nargs="+", help="Ignore specific repositories")
    parser.add_argument("--max-resolved", type=int, help="Maximum number of resolved instances")
    parser.add_argument("--exclude-instance-ids", nargs="+", help="Instance IDs to exclude")
    parser.add_argument("--response-format", 
                       choices=[format.value for format in LLMResponseFormat],
                       help="Response format for the model")
    parser.add_argument("--message-history", 
                       choices=[history.value for history in MessageHistoryType],
                       help="Message history type")
    
    args = parser.parse_args()
    
    # Create model settings
    model_settings = CompletionModel(
        model=args.model,
        temperature=0.0,
        max_tokens=3000,
        api_key=args.api_key,
        base_url=args.base_url,
        response_format=LLMResponseFormat(args.response_format) if args.response_format else None
    )
    
    # Create agent settings
    agent_settings = AgentSettings(
        completion_model=model_settings,
        message_history_type=MessageHistoryType(args.message_history) if args.message_history else MessageHistoryType.MESSAGES,
        system_prompt=None
    )
    
    # Create tree search settings
    tree_search_settings = TreeSearchSettings(
        max_iterations=args.max_iterations,
        max_expansions=args.max_expansions,
        max_cost=args.max_cost,
        model=model_settings,
        agent_settings=agent_settings
    )
    
    # Initialize repository
    repository = EvaluationFileRepository(os.getenv("MOATLESS_DIR", "./evals"))
    
    # Create evaluation name
    evaluation_name = create_evaluation_name(
        model=args.model,
        date=None,
        max_expansions=args.max_expansions,
        response_format=LLMResponseFormat(args.response_format) if args.response_format else None,
        message_history=MessageHistoryType(args.message_history) if args.message_history else None
    )
    
    # Create evaluation using factory
    evaluation = create_evaluation(
        repository=repository,
        evaluation_name=evaluation_name,
        settings=tree_search_settings,
        split=args.split,
        instance_ids=args.instance_ids,
        exclude_instance_ids=args.exclude_instance_ids,
        repos=args.repos,
        ignore_repos=args.ignore_repos,
        min_resolved=args.min_resolved,
        max_resolved=args.max_resolved
    )

    try:
        # Create event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Create monitor
        monitor = EvaluationMonitor(repository, evaluation)
        
        # Create runner with event handler
        runner = EvaluationRunner(
            repository=repository,
            evaluation=evaluation,
            dataset_name="princeton-nlp/SWE-bench_Lite",
            num_workers=args.num_workers,
            use_testbed=True
        )
        
        # Add event handler
        runner.add_event_handler(monitor.handle_event)
        
        # Create monitoring task
        monitoring_task = loop.create_task(monitor.start_monitoring())
        
        logger.info("Running evaluation")
        # Run evaluation in executor and wait for both tasks
        loop.run_until_complete(asyncio.gather(
            loop.run_in_executor(ThreadPoolExecutor(), runner.run_evaluation),
            monitoring_task
        ))
    except Exception as e:
        # Use rich to print error in red
        console = Console()
        console.print(f"[red]Error: {str(e)}")
        console.print("[red]Traceback:")
        console.print_exception()
        sys.exit(1)
    finally:
        if 'loop' in locals():
            loop.close()

if __name__ == "__main__":
    main()