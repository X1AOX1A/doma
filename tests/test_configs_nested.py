from pydantic import BaseModel, Field

from doma.configs import (
    get_config_field_recursively,
    build_config_from_flattened_dict,
)


class InnerConfig(BaseModel):
    x_value: int = Field(gt=0, description="Inner x value")
    y_ratio: float = Field(default=1.5, description="Inner y ratio")


class OuterConfig(BaseModel):
    name: str = Field(description="A name")
    inner: InnerConfig = Field(default=InnerConfig(x_value=1))


def test_get_config_field_recursively_flattens_nested_fields_in_order():
    # Default order should follow declaration: outer leafs first, then nested leafs
    fields = list(get_config_field_recursively(OuterConfig))
    names = [name for name, _ in fields]
    assert names == ["name", "x_value", "y_ratio"]


def test_get_config_field_recursively_reverse_order():
    # Reverse should process nested model first, then outer leafs
    fields = list(get_config_field_recursively(OuterConfig, reverse=True))
    names = [name for name, _ in fields]
    assert names == ["x_value", "y_ratio", "name"]


def test_build_config_from_flattened_dict_builds_nested_models():
    # Provide flattened keys without any prefix; nested model should be built accordingly
    flattened = {"name": "alice", "x_value": 10}
    cfg = build_config_from_flattened_dict(flattened, OuterConfig)
    assert isinstance(cfg, OuterConfig)
    assert cfg.name == "alice"
    assert cfg.inner.x_value == 10
    # Missing nested field uses default
    assert cfg.inner.y_ratio == 1.5


def test_build_config_from_flattened_dict_deeply_nested_models():
    class Level2(BaseModel):
        m: int = 1

    class Level1(BaseModel):
        z: int = 2
        l2: Level2 = Level2()

    class Root(BaseModel):
        a: int = 3
        l1: Level1 = Level1()

    flattened = {"a": 9, "z": 20, "m": 7}
    cfg = build_config_from_flattened_dict(flattened, Root)

    assert cfg.a == 9
    assert cfg.l1.z == 20
    assert cfg.l1.l2.m == 7

