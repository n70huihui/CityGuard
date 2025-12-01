from dataclasses import dataclass

@dataclass
class AgentCard:
    """
    AgentCard 类，用于模拟 A2A 协议，表示车辆的能力
    """
    car_id: str
    location: tuple[float, float]
    is_working: bool
    ability: list[str]
    speed: float