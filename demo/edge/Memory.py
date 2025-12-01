from collections import defaultdict

class Memory:
    """
    第三方存储，可以是 MEC，也可以是 Redis 等，这里基于内存使用 dict 进行简单模拟
    """
    def __init__(self):
        self.memory: dict[str, object] = defaultdict(object)
        self.list_memory: dict[str, list] = defaultdict(list)
        self.hash_memory: dict[str, dict] = defaultdict(dict)

    def get(self, key: str) -> object:
        return self.memory[key]

    def set(self, key: str, value: object) -> None:
        self.memory[key] = value

    def get_list(self, key: str) -> list:
        return self.list_memory[key]

    def rpush(self, key: str, value: object) -> None:
        self.list_memory[key].append(value)

    def set_list(self, key: str, value: list) -> None:
        self.list_memory[key] = value

    def get_hash(self, key: str) -> dict:
        return self.hash_memory[key]

    def set_hash(self, key: str, value: dict) -> None:
        self.hash_memory[key] = value