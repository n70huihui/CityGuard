from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate

"""
该文件存放整个 demo 项目所有的提示词信息
变量名统一为 function_name + [other_msg] + template
"""

parse_user_prompt_template = HumanMessagePromptTemplate.from_template("""
用户输入为: {user_input}
你需要把用户输入解析成位置地点，位置经纬度，以及任务信息
可以调用工具来根据位置返回对应经纬度。
""")

handle_image_observation_template = HumanMessagePromptTemplate.from_template("""
现有车辆观测图片，请你根据本次的任务：{task_description}，把图片内容总结成文本。
注意，仅总结和任务相关的信息。最终回复的文本不要包含 Markdown 形式，仅文字内容即可。
""")

handle_text_observation_template = HumanMessagePromptTemplate.from_template("""
现有车辆在任务路段位置: {task_location} 的观测信息: {observation}，请你根据本次的任务：{task_description}，总结出本次的观测结果以及对应的证据，形成报告。
此外，本次观测的其他额外信息: 任务 id: {task_id}, 车辆 id: {car_id}, 车辆朝向: {car_direction}, 观测时间: {observation_timestamp} 也一并纳入报告中。
在报告的总结中，请着重强调车辆的朝向以及对应的路段，可以把对应的 0-360 的角度转换为东南西北等方向来描述。
""")

multi_view_understanding_template = HumanMessagePromptTemplate.from_template("""
现有车辆报告信息: {simple_report_list}，以及你自己的报告信息: {self_report}。
请你根据本次的任务：{task_description}，参考别的车辆的报告，尝试检查并修正自己的结果。
注意，如果其他车辆的报告内容和你自己负责的报告内容，不一致，可以不参考。
例如：你自己的报告中观测的是某路段的东侧，车辆 A 的报告观测的是路段东侧，B 车辆的报告观测的是路段西侧，那么你可以参考 A 的报告，B 的就不需要关注了。
仅修正你自己报告信息中的 result 和 evidence 字段，其他字段不变。
如果你觉得别的车辆的总结报告不太好理解，必要时可以通过工具查看别的车辆的原始观测，但是注意不要每次都查，仅在必要的时候查阅即可。
返回最终的报告。
""")

summary_template = HumanMessagePromptTemplate.from_template("""
现有所有的车辆报告信息: {report_list}，以及本次任务的任务描述: {task_description}，任务 id: {task_id}。
请你总结本次任务的结果，并返回总结。在总结中需要注明路段信息，以及对应位置的情况。
""")