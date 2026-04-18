import os

from dotenv import load_dotenv

# 从.env文件加载环境变量
load_dotenv(verbose=True)

# 加载deepseek环境变量
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")

# 加载火山环境变量
ARK_API_KEY = os.getenv("ARK_API_KEY")
ARK_BASE_URL = os.getenv("ARK_BASE_URL")