from collections import defaultdict

class InMemoryStore:
    """
    第三方存储，可以是 MEC，也可以是 Redis 等，这里基于内存使用 dict 进行简单模拟
    """
    def __init__(self):
        self.list_memory: dict[str, list] = defaultdict(list)

    def get_list(self, key: str) -> list:
        return self.list_memory[key]

    def rpush(self, key: str, value: object) -> None:
        self.list_memory[key].append(value)

    def set_list(self, key: str, value: list) -> None:
        self.list_memory[key] = value