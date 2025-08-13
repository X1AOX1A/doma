from collections import deque
from dataclasses import dataclass
import gc
from multiprocessing import Process
from multiprocessing import Event as get_process_event
from multiprocessing.synchronize import Event as ProcessEvent
import os
import socket
import threading
from time import sleep, time
from typing import Literal, Optional

from loguru import logger
import torch

from doma.configs import ControllerConfig
from doma.core import Signal, SocketData, get_socket, recv_socket_data, send_socket_data


def compute_storage_size(gb):
    return int(gb * 1024 * 1024 * 1024 / 8)


@dataclass
class GPUSnapshot:
    used_mem: float
    free_mem: float
    util: float


class GPUController(Process):
    def __init__(self, id: int, config: ControllerConfig, stop_signal: ProcessEvent):
        """
        Args:
            id: GPU id
            config: Config
        """
        super().__init__()
        self.id = id
        self.config = config
        self.alg_config = config.alg_config
        self.device = torch.device(f"cuda:{self.id}")
        self.stop_signal = stop_signal

    def run(self):
        self.start_inspect()
        while not self.validate_hold_condition():
            sleep(1)
            if self.stop_signal.is_set():
                break
        if not self.stop_signal.is_set():
            logger.info(f"Start holding GPU {self.id}")
            self.hold()
        self.stop_inspect()

    def start_inspect(self):
        self.gpu_snapshot_queue = deque(maxlen=int(self.config.wait_minutes * 60))
        self.history_queue_lock = threading.Lock()
        self.inspect_stop_signal = threading.Event()
        self.inspect_executor = threading.Thread(
            target=self._inspect_worker, daemon=True
        )
        self.inspect_executor.start()

    def stop_inspect(self):
        self.inspect_stop_signal.set()
        self.inspect_executor.join()
        self.inspect_executor = None
        self.inspect_stop_signal = None
        self.gpu_snapshot_queue = None
        self.history_queue_lock = None

    def hold(self):
        holder = None
        operator = None

        gb = self.config.hold_mem
        util = self.config.hold_util
        if gb is None:
            gb = self.get_mem_total() * 0.5
        operator_size = int(compute_storage_size(self.alg_config.operator_gb) / 2)
        operator = torch.ones([operator_size], dtype=torch.double, device=self.device)
        max_sleep_time = self.alg_config.max_sleep_time
        min_sleep_time = self.alg_config.min_sleep_time
        mid_sleep_time = (max_sleep_time + min_sleep_time) / 2
        tic = time()
        inspect_interval = self.alg_config.inspect_interval
        util_samples = []
        util_samples_num = self.alg_config.util_samples_num
        find_target_sleep_time = False
        # adjust sleep time to keep the utilization at util using binary search
        first = True

        while not self.stop_signal.is_set():
            result = torch.mul(operator, operator)
            if first:
                used_gb = self.get_mem_used()
                holder_gb = gb - used_gb
                if holder_gb < 0:
                    raise ValueError(
                        f"Target GB ({gb}) is less than used GB ({used_gb}). Please reduce the operator GB ({self.alg_config.operator_gb})."
                    )
                holder_size = compute_storage_size(holder_gb)
                holder = torch.randn(
                    [holder_size], dtype=torch.double, device=self.device
                )  # noqa: F841
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
        with torch.device(f"cuda:{self.id}"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    def _inspect_worker(self):
        while not self.inspect_stop_signal.is_set():
            with self.history_queue_lock:
                self.gpu_snapshot_queue.append(
                    GPUSnapshot(
                        used_mem=self.get_mem_used(),
                        free_mem=self.get_mem_free(),
                        util=self.get_util(),
                    )
                )
            sleep(1)

    def get_mem_used(self):
        return torch.cuda.device_memory_used(self.device) / (1024**3)

    def get_mem_total(self):
        return torch.cuda.get_device_properties(self.device).total_memory / (1024**3)

    def get_mem_free(self):
        return self.get_mem_total() - self.get_mem_used()

    def get_util(self):
        return torch.cuda.utilization(device=self.device) / 100

    def get_history_metric(
        self,
        name: Literal["used_mem", "free_mem", "util"],
        metric_type: Literal["avg", "max", "min"],
    ):
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

    def validate_hold_condition(self):
        return (
            self.is_history_full()
            and self.get_history_metric("used_mem", "max") < self.config.mem_threshold
        )


class GPUGroupManager:
    def __init__(self, config: ControllerConfig):
        self.config = config
        self.launch_socket()
        self.gpu_controllers: list[GPUController] = []
        self.controller_stop_signal = get_process_event()

    def reset_controllers(self):
        self.stop_controllers()
        self.gpu_controllers = []
        self.controller_stop_signal.clear()
        for i in range(torch.cuda.device_count()):
            controller = GPUController(i, self.config, self.controller_stop_signal)
            self.gpu_controllers.append(controller)

    def launch_socket(self):
        self.socket, self.server_address = get_socket()
        if os.path.exists(self.server_address):
            raise FileExistsError(
                f"Socket file {self.server_address} already exists. doma may be already running or the previous instance is not shutdown properly."
            )
        os.makedirs(os.path.dirname(self.server_address), exist_ok=True)
        self.socket.bind(self.server_address)
        self.socket.listen()

    def start_controllers(self):
        self.controller_stop_signal.clear()
        for controller in self.gpu_controllers:
            controller.start()

    def stop_controllers(self):
        self.controller_stop_signal.set()
        for controller in self.gpu_controllers:
            logger.info(f"Stopping controller {controller.id}")
            if controller.is_alive():
                controller.join()

    def update_config(self, config: Optional[ControllerConfig]):
        if config is not None:
            self.config = config

    def test_address_alive(self):
        if not os.path.exists(self.server_address):
            logger.error(
                f"Socket file {self.server_address} does not exist. doma may be shutdown unexpectedly."
            )
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
                logger.info(f"Received signal: {signal}")
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
                            self.stop_controllers()
                            run = False
                        case Signal.GREETING:
                            pass
                except Exception as e:
                    error = e
                send_socket_data(conn, SocketData(signal=Signal.GREETING, error=error))
        self.socket.close()
        os.remove(self.server_address)
