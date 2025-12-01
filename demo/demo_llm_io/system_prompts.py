from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate

"""
该文件存放整个 demo 项目所有的提示词信息
变量名统一为 function_name + [other_msg] + template
"""

parse_user_prompt_template = HumanMessagePromptTemplate.from_template("用户输入为: {user_input}")