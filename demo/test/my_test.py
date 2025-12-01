import uuid, json, heapq

from demo.demo_settings.variables import *
from demo.vehicle.AgentCard import AgentCard

target_location = (0, 0)

agent_cards = [vehicle.get_agent_card() for vehicle in vehicle_list]

agent_card_models = [AgentCard(**json.loads(agent_card)) for agent_card in agent_cards]
print(agent_card_models)

# (距离, 速度) 优先短距离，慢速度
scores = []
score_heap = []
for agent_card in agent_card_models:
    distance = (agent_card.location[0] - target_location[0]) ** 2 + (agent_card.location[1] - target_location[1]) ** 2
    score = -distance * 0.7 - agent_card.speed * 0.3
    scores.append(-score)
    heapq.heappush(score_heap, score)
    if len(score_heap) > 3:
        heapq.heappop(score_heap)


print(sorted(scores))

while score_heap:
    score = heapq.heappop(score_heap)
    print(-score)