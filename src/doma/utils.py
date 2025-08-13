from doma.core import get_socket, socket_timeout, send_socket_data, recv_socket_data, SocketData, SOCKET_TIMEOUT
from pydantic import BaseModel
from typing import Callable, Type
import click
from functools import update_wrapper
from doma.configs import build_config_from_flattened_dict, get_config_field_recursively

def exchange_socket_data(data:SocketData,timeout:float=SOCKET_TIMEOUT) -> SocketData:
    socket,addr = get_socket()
    try:
        with socket_timeout(socket, timeout):
            socket.connect(addr)
        send_socket_data(socket, data, timeout)
        result = recv_socket_data(socket, timeout)
    except Exception as e:
        raise e
    finally:
        socket.close()
    return result

def show_flattened_config(config:BaseModel) -> str:
    config_dict = config.model_dump()
    flattened_dict = {}
    def _flatten_dict(d):
        for name,value in d.items():
            if isinstance(value,dict):
                _flatten_dict(value)
            else:
                flattened_dict[name] = value
    _flatten_dict(config_dict)
    return "\n".join([f"{name}: {value}" for name,value in flattened_dict.items()])

def cfg_as_opt(config_cls:Type[BaseModel]):
    def decorator(func:Callable):
        def wrapper(**kwargs):
            config = build_config_from_flattened_dict(kwargs,config_cls)
            return func(config)

        for name,field in get_config_field_recursively(config_cls,reverse=True):
            name = name.replace("_","-")
            if field.is_required():
                wrapper = click.option(f"--{name}", type=field.annotation, help=field.description)(wrapper)
            else:
                wrapper = click.option(f"--{name}", type=field.annotation, default=field.default, help=field.description)(wrapper)
        update_wrapper(wrapper,func)
        return wrapper
    
    return decorator