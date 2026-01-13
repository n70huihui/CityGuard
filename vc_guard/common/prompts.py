from langchain_core.prompts import HumanMessagePromptTemplate

"""
该文件存放整个 demo 项目所有的提示词信息
变量名统一为 function_name + [other_msg] + template
"""

parse_user_prompt_template = HumanMessagePromptTemplate.from_template("""
用户输入为: {user_input}
你需要把用户输入解析成位置地点，位置经纬度，以及任务信息
可以调用工具来根据位置返回对应经纬度。
""")

get_best_vehicle_id_list_template = HumanMessagePromptTemplate.from_template("""
你是一个智能车辆调度规划师，需要为拍摄任务在地图中选择最优的车辆。
## 任务信息
当前地图信息（0 表示障碍，1 表示通路，其余数字表示道路拥堵程度，数字越大拥堵程度越高）: {grid_matrix}
拍摄任务地点坐标：{task_location}
需要选择的车辆数量：{num_of_vehicles}
可用车辆列表: {agent_card_models}
## 筛选规则
1. 优先选择距离任务地点近的车辆；
2. 其次选择速度适中的车辆（无需过快，避免到达后等待过久，也无需过慢，避免延误）；
3. 仅从提供的可用车辆中选择，禁止虚构车辆ID；
4. 在工作中的车辆不要选择；
5. 你不需要真正地计算车辆的最优路径，你只需要大致根据上述规则进行评估即可，真正的路径计算不需要你做。
6. 返回最终的车辆 id 列表，数量需要和需要选择的车辆一致，如果当前车辆数量总数少于需要选择的车辆数量，则返回当前所有车辆 id 列表。如果没有可用车辆，直接返回空列表即可。
""")

handle_image_observation_template = HumanMessagePromptTemplate.from_template("""
现有车辆观测图片，请你根据本次的任务：{task_description}，把图片内容总结成文本。
注意，仅总结和任务相关的信息。最终回复的文本不要包含 Markdown 形式，仅文字内容即可。
""")

handle_text_observation_template = HumanMessagePromptTemplate.from_template("""
现有车辆在任务路段位置: {task_location} 的观测信息。请你根据本次的任务：{task_description}，总结出本次的观测结果以及对应的证据，形成报告。
此外，本次观测的其他额外信息: 任务 id: {task_id}, 车辆 id: {car_id}, 车辆朝向: {car_direction}, 观测时间: {observation_timestamp} 也一并纳入报告中。
在报告的总结中，请着重强调车辆的朝向以及对应的路段，可以把对应的 0-360 的角度转换为东南西北等方向来描述。
""")

multi_view_understanding_template = HumanMessagePromptTemplate.from_template("""
现有车辆报告信息: {simple_report_list}，以及你自己的报告信息: {self_report}。
请你根据本次的任务：{task_description}，参考别的车辆的报告，尝试检查并修正自己的结果。
注意，如果其他车辆的报告内容和你自己负责的报告内容，不一致，可以不参考。
例如：你自己的报告中观测的是某路段的东侧，车辆 A 的报告观测的是路段东侧，B 车辆的报告观测的是路段西侧，那么你可以参考 A 的报告，B 的就不需要关注了。
仅修正你自己报告信息中的 result 和 evidence 字段，其他字段不变。
返回最终的报告。
""")

summary_template = HumanMessagePromptTemplate.from_template("""
现有所有的车辆报告信息: {report_list}，以及本次任务的任务描述: {task_description}，任务 id: {task_id}。
请你总结本次任务的结果，并返回总结。在总结中需要注明路段信息，以及对应位置的情况。
""")

multi_view_understanding_summary_template = HumanMessagePromptTemplate.from_template("""
现有车辆报告信息: {simple_report_list}。任务 id: {task_id}。
请你根据本次的任务：{task_description}，参考这些车辆的报告，进行多视角理解。例如：车辆 A、C 的报告显示路段东侧有违停情况，B 显示路段东侧看不到任何信息，那么你应该参考 A、C 的报告，B 的就不需要关注了。
返回最终的报告。在总结中需要注明路段信息，以及对应位置的情况。
""")

llm_judge_template = HumanMessagePromptTemplate.from_template("""
你是根因分析有效性检测工具，仅需输出 "CONTINUE" 或 "STOP"，无任何额外文字、标点或解释。
## 被检测的任务描述: {task_description}
## 检测规则
1. 有效根因分析（输出 STOP）：报告中指出了形成问题的原因
2. 无效根因分析（输出 CONTINUE）：
   - 无问题形成的根因分析；
   - 根因分析存在明显逻辑错误。

## 待检测内容
{final_report}
""")

evaluate_summary_template = HumanMessagePromptTemplate.from_template("""
你是计算机领域专注于城市市容管理智能检测的学术评审专家，精通目标检测结果的语义一致性与有效性评估。
你的核心任务是基于以下明确标准和目标检测场景，对智能体的检测结果进行客观量化评分，仅输出0.0-10.0的浮点分数（保留1位小数），不添加任何额外文本、格式或解释。

## 1. 目标检测场景定义举例（必须严格遵循）
- garbage：违规场景为“路段周围垃圾违规堆放造成异味”，核心判定要素：是否存在垃圾堆积（含溢出垃圾桶、散落地面）、垃圾是否与异味存在直接关联；
- illegal_parking：违规场景为“违规停车造成车辆拥堵”，核心判定要素：是否存在违规停车行为、违规停车是否与车辆拥堵存在直接关联。

## 2. 评估维度与量化标准（总分10.0分）
### 维度1：准确性（4.0分）
- 4.0分：检测结论与场景定义完全一致（存在违规则明确对应核心要素，不存在违规则判断依据充分）；
- 2.0-3.5分：检测结论基本符合场景定义，但核心要素描述模糊（如未明确垃圾堆积位置、未说明违规停车与拥堵的关联）；
- 0.0-1.5分：检测结论与场景定义不符（如垃圾场景误判为其他污染源、违规停车场景未识别拥堵关联）或存在明显错误。

### 维度2：完整性（3.0分）
- 3.0分：完整包含核心要素的关键信息（垃圾场景：位置+垃圾类型+异味关联；违规停车场景：位置+停车数量+拥堵影响）；
- 1.5-2.5分：包含部分关键信息（如仅说明存在垃圾堆积，未提及异味来源）；
- 0.0-1.0分：未包含核心要素的关键信息（如仅说“存在异味”，未提及垃圾）。

### 维度3：相关性（2.0分）
- 2.0分：描述完全围绕场景核心要素，无无关信息（如垃圾场景不冗余提及车辆排气、工厂排放等非核心内容）；
- 1.0分：描述以核心要素为主，但包含少量无关信息；
- 0.0分：描述重点偏离场景核心要素，无关信息占比超50%。

### 维度4：明确性（1.0分）
- 1.0分：结论清晰明确（明确“存在”或“不存在”违规，无模糊表述）；
- 0.0分：结论模糊（如“可能存在异味”“疑似违规停车”等不确定表述）。

## 3. 评分规则
1. 先根据检测结果的“违规存在状态”（存在/不存在），对应场景核心要素逐项匹配，分别计算四个维度得分，总分=各维度得分之和；
2. 若检测结果未提及场景核心要素（如垃圾场景未提“垃圾”，违规停车场景未提“违规停车”），直接判0.0分；
3. 分数保留1位小数，如9.5、3.0、0.5（禁止整数形式如10，需写10.0）。

已知目标检测场景：{target_task}
智能体检测结果：{summary}
""")