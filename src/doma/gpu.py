import os
from typing import Literal,Optional
import torch
from time import sleep,time
from doma.core import get_socket,Signal,recv_socket_data,send_socket_data, SocketData
from doma.configs import ControllerConfig
import threading
from collections import deque
from dataclasses import dataclass
from loguru import logger
import socket
import gc
def compute_storage_size(gb):
    return int(gb *1024 * 1024 * 1024 / 8)


@dataclass
class GPUSnapshot:
    used_mem:float
    free_mem:float
    util:float


class GPUController:
    def __init__(self, id:int, config:ControllerConfig):
        """
        Args:
            id: GPU id
            config: Config
        """
        self.id = id
        self.config = config
        self.alg_config = config.alg_config
        self.device = torch.device(f'cuda:{self.id}')
        self.stop_signal = threading.Event()
        self.holding_executor = None
        self.holding = False
        # Maintain the last wait seconds' timestamp
        self.gpu_snapshot_queue = deque(maxlen=int(config.wait_minutes * 60))
        self.history_queue_lock = threading.Lock()
        self.inspect_stop_signal = False
        self.inspect_executor = threading.Thread(target=self.inspect,daemon=True)
        self.inspect_executor.start()

    def hold(self):

        holder = None
        operator = None

        gb = self.config.hold_mem
        util = self.config.hold_util
        if gb is None:
            gb = self.get_mem_total() * 0.5
        operator_size = int(compute_storage_size(self.alg_config.operator_gb)/2)
        operator = torch.ones([operator_size], dtype=torch.double, device=self.device)
        max_sleep_time = self.alg_config.max_sleep_time
        min_sleep_time = self.alg_config.min_sleep_time
        mid_sleep_time = (max_sleep_time + min_sleep_time) / 2
        tic = time()
        inspect_interval = self.alg_config.inspect_interval
        util_samples = []
        util_samples_num = self.alg_config.util_samples_num
        find_target_sleep_time = False
        #adjust sleep time to keep the utilization at util using binary search
        first = True

        while not self.stop_signal.is_set():
            result = torch.mul(operator, operator)
            if first:
                used_gb = self.get_mem_used()
                holder_gb = gb - used_gb
                if holder_gb < 0:
                    raise ValueError(f"Target GB ({gb}) is less than used GB ({used_gb}). Please reduce the operator GB ({self.alg_config.operator_gb}).")
                holder_size = compute_storage_size(holder_gb)
                holder = torch.randn([holder_size], dtype=torch.double, device=self.device)  # noqa: F841
                tic = time()
                first = False
                continue
            toc = time()
            if not find_target_sleep_time and toc - tic >= inspect_interval:
                if len(util_samples) < util_samples_num:
                    util_samples.append(self.get_util())
                    sleep(mid_sleep_time)
                    tic = time()
                    continue
                cur_util = sum(util_samples) / util_samples_num
                util_samples.clear()
                if abs(cur_util - util) <= self.alg_config.util_eps:
                    find_target_sleep_time = True
                    continue
                if cur_util < util:
                    max_sleep_time = mid_sleep_time
                elif cur_util > util:
                    min_sleep_time = mid_sleep_time
                mid_sleep_time = (max_sleep_time + min_sleep_time) / 2
                tic = time()
            sleep(mid_sleep_time)
        
        if holder is not None:
            del holder
        if operator is not None:
            del operator
        if result is not None:
            del result
        gc.collect()
        with torch.device(f'cuda:{self.id}'):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    def inspect(self):
        while not self.inspect_stop_signal:
            with self.history_queue_lock:
                self.gpu_snapshot_queue.append(GPUSnapshot(
                    used_mem=self.get_mem_used(),
                    free_mem=self.get_mem_free(),
                    util=self.get_util()
                ))
            sleep(1)
    

    def get_mem_used(self):
        return torch.cuda.device_memory_used(self.device) / (1024**3)
    
    def get_mem_total(self):
        return torch.cuda.get_device_properties(self.device).total_memory / (1024**3)
    
    def get_mem_free(self):
        return self.get_mem_total() - self.get_mem_used()
    
    def get_util(self):
        return torch.cuda.utilization(device=self.device) / 100
    
    def start_holding(self):
        if self.holding_executor is not None:
            raise RuntimeError("GPUHolder is already running")
        self.holding_executor = threading.Thread(target=self.hold,daemon=True)
        self.stop_signal.clear()
        self.holding_executor.start()
        self.holding = True
        
    def stop_holding(self):
        if self.holding_executor is None:
            raise RuntimeError("GPUHolder is not running")
        self.stop_signal.set()
        self.holding_executor.join()
        self.holding_executor = None
        self.holding = False

    def get_history_metric(self,name:Literal["used_mem","free_mem","util"],metric_type:Literal["avg","max","min"]):
        with self.history_queue_lock:
            metrics = [getattr(snapshot, name) for snapshot in self.gpu_snapshot_queue]
            if metric_type == "avg":
                return sum(metrics) / len(metrics)
            elif metric_type == "max":
                return max(metrics)
            elif metric_type == "min":
                return min(metrics)
    
    def is_history_full(self):
        with self.history_queue_lock:
            return len(self.gpu_snapshot_queue) == self.gpu_snapshot_queue.maxlen
    
    def reset_history(self):
        with self.history_queue_lock:
            self.gpu_snapshot_queue.clear()

class GPUGroupManager:
    def __init__(self, config: ControllerConfig):
        self.config = config
        self.launch_socket()
        self.running_signal = threading.Event()
        self.running_signal.clear()
        self.running_thread = None
        self.gpu_controllers = None

    def reset_controllers(self):
        if self.running_signal.is_set():
            # stop controllers
            self.stop_controllers()
        self.gpu_controllers = [
            GPUController(i, self.config) for i in range(torch.cuda.device_count())
        ]
    

    def launch_socket(self):
        self.socket, self.server_address = get_socket()
        if os.path.exists(self.server_address):
            raise FileExistsError(f"Socket file {self.server_address} already exists. doma may be already running or the previous instance is not shutdown properly.")
        self.socket.bind(self.server_address)
        self.socket.listen()

    def _validate_controller_start_condition(self, controller: GPUController):
        return (
            controller.is_history_full()
            and not controller.holding
            and controller.get_history_metric("used_mem", "max")
            < self.config.mem_threshold
        )

    def _controllers_loop(self):
        # reset history
        for controller in self.gpu_controllers:
            controller.reset_history()

        while self.running_signal.is_set():
            for controller in self.gpu_controllers:
                if self._validate_controller_start_condition(controller):
                    controller.start_holding()

            sleep(1)
        # stop all controllers
        for controller in self.gpu_controllers:
            if controller.holding:
                controller.stop_holding()
            controller.inspect_stop_signal = True

    def start_controllers(self):
        if self.running_thread is not None:
            raise RuntimeError(
                "Controllers are already running. Please stop them first."
            )
        self.running_signal.set()
        self.running_thread = threading.Thread(target=self._controllers_loop,daemon=True)
        self.running_thread.start()

    def stop_controllers(self):
        if self.running_thread is None:
            raise RuntimeError("Controllers are not running. Please start them first.")
        self.running_signal.clear()
        self.running_thread.join()
        self.running_thread = None

    
    def update_config(self,config:Optional[ControllerConfig]):
        if config is not None:
            self.config = config
    
    def test_address_alive(self):
        if not os.path.exists(self.server_address):
            logger.error(f"Socket file {self.server_address} does not exist. doma may be shutdown unexpectedly.")
            return False
        return True
    
    def listen_signal(self):
        run = True
        while run and self.test_address_alive():
            self.socket.settimeout(10)
            try:
                conn, _ = self.socket.accept()
            except socket.timeout:
                continue
            with conn:
                socket_data = recv_socket_data(conn)
                signal = socket_data.signal
                config = socket_data.config
                error = None
                try:
                    match signal:
                        case Signal.START:
                            self.update_config(config)
                            self.reset_controllers()
                            self.start_controllers()
                        case Signal.STOP:
                            self.stop_controllers()
                        case Signal.RESTART:
                            self.update_config(config)
                            self.reset_controllers()
                            self.start_controllers()
                        case Signal.SHUTDOWN:
                            if self.running_thread is not None:
                                self.stop_controllers()
                            run = False
                        case Signal.GREETING:
                            pass
                except Exception as e:
                    error = str(e)
                send_socket_data(conn, SocketData(signal=Signal.GREETING, error=error))
        self.socket.close()
        os.remove(self.server_address)