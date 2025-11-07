import abc
from dataclasses import dataclass

import draccus


@dataclass
class MotorsBusConfig(draccus.ChoiceRegistry, abc.ABC):
    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)


@MotorsBusConfig.register_subclass("dynamixel")
@dataclass
class DynamixelMotorsBusConfig(MotorsBusConfig):
    port: str
    motors: dict[str, tuple[int, str]]
    mock: bool = False


@MotorsBusConfig.register_subclass("feetech")
@dataclass
class FeetechMotorsBusConfig(MotorsBusConfig):
    port: str
    motors: dict[str, tuple[int, str]]
    mock: bool = False


@MotorsBusConfig.register_subclass("piper")
@dataclass
class PiperMotorsBusConfig(MotorsBusConfig):
    can_name: str
    motors: dict[str, tuple[int, str]]

#主从臂模式下的piper机械臂驱动接口
@MotorsBusConfig.register_subclass("MSpiper")
@dataclass
class MSPiperMotorsBusConfig(MotorsBusConfig):
    can_name: str
    slaver_motors: dict[str, tuple[int, str]]
    master_motors: dict[str, tuple[int, str]]