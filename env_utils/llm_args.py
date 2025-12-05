import os
from dotenv import load_dotenv

# 加载.env文件
load_dotenv()

# 读取环境变量
api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")
model = os.getenv("MODEL")
visual_model = os.getenv("VISUAL_MODEL")