from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from env_utils.llm_args import *
from langchain.agents import create_agent

from guard.common.prompt import verifier_sys_prompt

verifier = create_agent(
    model=ChatOpenAI(model=visual_model, base_url=base_url, api_key=api_key),
    tools=[],
    system_prompt=verifier_sys_prompt.format()
)

def verify(report: str, answer: str) -> float:
    response = verifier.invoke(
        {"messages": [HumanMessage(content=f"智能体报告结果如下：{report}; 参考答案如下：{answer}")]},
    )
    return float(response["messages"][-1].content_blocks[-1]['text'])

if __name__ == "__main__":
    print(verify(report="""车载视角分析结果明确指向 **area_1 东边 road_1_1 上的垃圾堆积问题**：

- `area_1_camera_1`（位于 area_1 东边、紧邻 road_1_1）画面显示：  
  ✅ **大量垃圾堆积在绿色垃圾桶旁，部分散落在草坪和人行道上**；  
  ✅ 分析报告指出：“可能产生刺鼻异味，与市民举报内容**高度相关**，是根因分析的重点对象”。

- 对比 `area_1_camera_5`（东北角）仅垃圾桶满溢、环境整洁，异味可能性低，可排除为主因。

### 🔍 根因确认与逻辑闭环：
- 垃圾长期堆积 → 有机物腐败发酵 → 释放硫化氢、氨气等刺鼻气体；  
- 该点位**正位于 road_1_1 路段上（area_1 东边）**，与市民举报地点完全一致；  
- 监控 `monitor_1` 未拍到此场景，因其视角聚焦街角主干道与车辆通行区，而垃圾堆位于**人行道旁草坪边缘**（监控盲区），解释了“有味无象”的矛盾；  
- 地理位置高度吻合：area_1 与 road_1_1 直接毗邻，气味极易沿路面扩散，被路人感知。

### ✅ 结论：
**根因已定位：area_1 东侧 road_1_1 人行道旁存在大量未清理的生活垃圾堆积，导致刺鼻异味。**

### 📌 建议处置：
- 立即调度环卫部门清理该处垃圾堆及周边散落物；  
- 检查该区域垃圾桶配置密度与清运频次，避免再次满溢；  
- 巡查 area_1 内其他角落是否存在类似隐患（如后巷、隐蔽角落）；  
- （可选）后续可调取 `area_2` 车载视角作交叉验证，但当前证据已充分，非紧急必要。

无需进一步调取其他监控或车载视角。""", answer="area_1区域内垃圾违规堆放，导致异味扩散至road_1_1"))