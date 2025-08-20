import os
import signal
from time import sleep

import click
import psutil

from doma.configs import ControllerConfig, LaunchConfig
from doma.core import PID_PATH, Signal, SocketData, daemonize
from doma.gpu import GPUGroupManager
from doma.utils import (
    cfg_as_opt,
    exchange_socket_data,
    get_logger,
    is_server_dead,
    show_flattened_config,
)

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.version_option()
@click.group(context_settings=CONTEXT_SETTINGS)
def cli():
    """
    Doma is a tool for holding idle GPU resources.
    """
    pass


def _status() -> bool:
    """
    Show the status of doma server.
    """
    try:
        socket_data = exchange_socket_data(SocketData(signal=Signal.GREETING))
        if socket_data.error is not None:
            raise socket_data.error
        return True, None
    except Exception as e:
        return False, e


@cli.command()
def status():
    """
    Show the status of doma server.
    """
    logger = get_logger()
    is_running, error = _status()
    if is_running:
        logger.info("Server is healthy.")
    else:
        logger.error(f"Cannot connect to server. {error}")


@cli.command()
@cfg_as_opt(LaunchConfig)
def launch(config: LaunchConfig):
    """
    Launch the doma server.
    """
    logger = get_logger()
    parent_dir = config.cache_path
    os.makedirs(parent_dir, exist_ok=True)
    log_path = f"{config.cache_path}/doma.log"
    if not is_server_dead(wait_time=1):
        logger.warning("Server is already running.")
        return

    def _launch_manager():
        import signal

        def signal_handler(signum, frame):
            """Handle shutdown signals gracefully"""
            logger = get_logger()
            logger.info(f"Received signal {signum}, shutting down...")
            # The gpu_group_manager.listen_signal() loop will handle cleanup

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        pid = os.getpid()
        with open(PID_PATH, "w") as f:
            f.write(str(pid))

        gpu_group_manager = GPUGroupManager(ControllerConfig())
        gpu_group_manager.listen_signal()
        os.remove(PID_PATH)

    daemonize(_launch_manager, stdout=log_path, stderr=log_path)

    MAX_RETRY = 10
    for _ in range(MAX_RETRY):
        sleep(1)
        is_running, error = _status()
        if is_running:
            logger.success("Server launched.")
            break
    else:
        logger.error(f"Failed to launch server. {str(error)}")


@cli.command()
@cfg_as_opt(ControllerConfig)
def start(config: ControllerConfig):
    """
    Start to wait for GPUs being idle and hold them with the given config.
    """
    logger = get_logger()
    if is_server_dead(wait_time=1):
        logger.warning("Server is not running. Run `doma launch` to launch the server.")
        return
    try:
        socket_data = exchange_socket_data(
            SocketData(signal=Signal.START, config=config)
        )
        if socket_data.error is not None:
            logger.exception(socket_data.error)
            raise RuntimeError(socket_data.error)
        logger.success(
            f"Service started with config: \n{show_flattened_config(config)}"
        )
    except Exception as e:
        logger.exception(e)
        logger.error("Failed to start service.")


@cli.command()
@cfg_as_opt(ControllerConfig)
def restart(config: ControllerConfig):
    """
    Release all GPUs and wait to hold them with the given config from the beginning. The behavior is the same as `start`.
    """
    logger = get_logger()
    if is_server_dead(wait_time=1):
        logger.warning("Server is not running. Run `doma launch` to launch the server.")
        return
    try:
        socket_data = exchange_socket_data(
            SocketData(signal=Signal.RESTART, config=config)
        )
        if socket_data.error is not None:
            raise RuntimeError(socket_data.error)
        logger.success(
            f"Service restarted with config: \n{show_flattened_config(config)}"
        )
    except Exception as e:
        logger.exception(e)
        logger.error("Failed to restart service.")


@cli.command()
def stop():
    """
    Stop holding and release all GPUs.
    """
    logger = get_logger()
    if is_server_dead(wait_time=1):
        logger.warning("Server is not running. Run `doma launch` to launch the server.")
        return
    try:
        socket_data = exchange_socket_data(SocketData(signal=Signal.STOP))
        if socket_data.error is not None:
            raise RuntimeError(socket_data.error)
        logger.success("Service stopped.")
    except Exception as e:
        logger.exception(e)
        logger.error("Failed to stop service.")


@cli.command()
def shutdown():
    """
    Shutdown the doma server.
    """
    logger = get_logger()
    if not os.path.exists(PID_PATH):
        logger.warning("Server is not running.")
        return
    with open(PID_PATH, "r") as f:
        pid = int(f.read())
    if not psutil.pid_exists(pid):
        os.remove(PID_PATH)
        logger.warning("Server is not running.")
        return

    try:
        socket_data = exchange_socket_data(SocketData(signal=Signal.SHUTDOWN))
        if socket_data.error is not None:
            raise socket_data.error
    except Exception as e:
        logger.warning(
            f"Failed to shutdown server gracefully: {e}.\nTrying to shutdown forcefully..."
        )
    finally:
        if is_server_dead(remove_pid_file_if_dead=True):
            logger.success("Server shutdown successfully.")
            return

        os.kill(pid, signal.SIGTERM)
        if is_server_dead(remove_pid_file_if_dead=True):
            logger.success("Server shutdown successfully.")
            return

        os.kill(pid, signal.SIGKILL)
        if is_server_dead(remove_pid_file_if_dead=True):
            logger.success("Server shutdown successfully.")
            return

        logger.error(
            f"Failed to shutdown server forcefully. Process {pid} is still running."
        )


if __name__ == "__main__":
    cli()
