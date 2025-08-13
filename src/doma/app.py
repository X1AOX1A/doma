import os
import sys
from time import sleep

import click

from doma.configs import ControllerConfig, LaunchConfig
from doma.core import Signal, SocketData, daemonize
from doma.gpu import GPUGroupManager
from doma.utils import cfg_as_opt, exchange_socket_data, show_flattened_config


def get_logger():
    from loguru import logger

    logger.remove()
    logger.add(sys.stdout, format="{message}", level="INFO")
    return logger


@click.group()
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
        logger.info("Server is running.")
    else:
        if isinstance(error, RuntimeError):
            logger.info("Server is not running.")
        else:
            logger.info(f"Server is not running. {error}")


@cli.command()
@cfg_as_opt(LaunchConfig)
def launch(config: LaunchConfig):
    """
    Launch the doma server.
    """
    log_path = config.log_path
    parent_dir = os.path.dirname(log_path)
    os.makedirs(parent_dir, exist_ok=True)

    def _launch_manager():
        gpu_group_manager = GPUGroupManager(ControllerConfig())
        gpu_group_manager.listen_signal()
        sys.exit(0)

    daemonize(_launch_manager, stdout=log_path, stderr=log_path)
    logger = get_logger()
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
    try:
        socket_data = exchange_socket_data(
            SocketData(signal=Signal.START, config=config)
        )
        if socket_data.error is not None:
            raise RuntimeError(socket_data.error)
        logger.success(
            f"Service started with config: \n{show_flattened_config(config)}"
        )
    except Exception as e:
        logger.error(f"Failed to start service: {e}.")


@cli.command()
@cfg_as_opt(ControllerConfig)
def restart(config: ControllerConfig):
    """
    Release all GPUs and wait to hold them with the given config from the beginning. The behavior is the same as `start`.
    """
    logger = get_logger()
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
        logger.error(f"Failed to restart service: {e}.")


@cli.command()
def stop():
    """
    Stop holding and release all GPUs.
    """
    logger = get_logger()
    try:
        socket_data = exchange_socket_data(SocketData(signal=Signal.STOP))
        if socket_data.error is not None:
            raise RuntimeError(socket_data.error)
        logger.success("Service stopped.")
    except Exception as e:
        logger.error(f"Failed to stop service: {e}.")


@cli.command()
def shutdown():
    """
    Shutdown the doma server.
    """
    logger = get_logger()
    try:
        socket_data = exchange_socket_data(SocketData(signal=Signal.SHUTDOWN))
        if socket_data.error is not None:
            raise RuntimeError(socket_data.error)
        logger.success("Server shutdown.")
    except FileNotFoundError:
        logger.warning("Server is not running.")
    except Exception as e:
        logger.error(f"Failed to shutdown server: {e}.")


if __name__ == "__main__":
    cli()
