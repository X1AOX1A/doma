from pydantic import BaseModel, Field, ConfigDict
from pydantic.fields import FieldInfo
import datetime
from typing import Type,Tuple,Dict,Any,Optional
class AlgorithmConfig(BaseModel):
    operator_gb: float = Field(default=1.0,gt=0,description="Operator size in GB for convergence control (smaller = higher precision but slower)")
    util_eps: float = Field(default=0.01,gt=0,description="Utilization epsilon for convergence precision (smaller = higher precision but slower). Defaults to 1%%")
    max_sleep_time: float = Field(default=1,gt=0,description="Maximum sleep time in seconds")
    min_sleep_time: float = Field(default=0,ge=0,description="Minimum sleep time in seconds")
    inspect_interval: float = Field(default=1,gt=0,description="Interval in seconds to inspect GPU utilization")
    util_samples_num: int = Field(default=5,gt=0,description="Number of samples to take for utilization")

class ControllerConfig(BaseModel):
    wait_minutes: float = Field(default=10,ge=0.1,description="Minutes to wait before holding GPU")
    mem_threshold: float = Field(default=0.01,gt=0,description="Memory threshold in GB")
    hold_mem: Optional[float] = Field(default=None,gt=0,description="Memory to hold in GB. Defaults to 50%% of free memory")
    hold_util: float = Field(default=0.5,gt=0,lt=1,description="GPU utilization to maintain (0-1)")
    alg_config: AlgorithmConfig = Field(default=AlgorithmConfig(),description="Algorithm configuration")


class LaunchConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    log_path: str = Field(default=f"/tmp/doma/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log",description="Path to log file")

def get_config_field_recursively(config:Type[BaseModel],reverse:bool=False) -> Tuple[str,FieldInfo]:
    if reverse:
        iterator = reversed(config.model_fields.items())
    else:
        iterator = config.model_fields.items()
    for name,field in iterator:
        if issubclass(field.annotation,BaseModel):
            yield from get_config_field_recursively(field.annotation)
        else:
            yield name,field

def build_config_from_flattened_dict(flattened_dict:Dict[str,Any],config_cls:Type[BaseModel]) -> BaseModel:
    config = config_cls.model_validate(flattened_dict)
    for name,field in config_cls.model_fields.items():
        if issubclass(field.annotation,BaseModel):
            setattr(config,name,build_config_from_flattened_dict(flattened_dict,field.annotation))
    return config