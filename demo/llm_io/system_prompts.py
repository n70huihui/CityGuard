from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate

"""
该文件存放整个 demo 项目所有的提示词信息
变量名统一为 function_name + [other_msg] + template
"""

parse_user_prompt_template = HumanMessagePromptTemplate.from_template("用户输入为: {user_input}")

handle_observation_template = HumanMessagePromptTemplate.from_template("""
现有车辆观测信息: {observation}，请你根据本次的任务：{task_description}，总结出本次的观测结果以及对应的证据，形成报告。
此外，本次观测的其他额外信息: 任务 id: {task_id}, 车辆 id: {car_id}, 观测时间: {observation_timestamp} 也一并纳入报告中。
""")