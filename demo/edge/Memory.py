from collections import defaultdict

class Memory:
    """
    第三方存储，可以是 MEC，也可以是 Redis 等，这里基于内存使用 dict 进行简单模拟
    """
    def __init__(self):
        self.memory: dict[str, object] = defaultdict(object)

    def get(self, key: str) -> object:
        return self.memory[key]

    def set(self, key: str, value: object) -> None:
        self.memory[key] = value