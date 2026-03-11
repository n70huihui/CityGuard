"""
CityGuard FastAPI Web 服务入口
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from guard.server.router import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时
    print("CityGuard 服务启动中...")
    yield
    # 关闭时
    print("CityGuard 服务关闭中...")


app = FastAPI(
    title="CityGuard API",
    description="城市异常检测智能体 Web 服务",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 中间件配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(router)


@app.get("/")
async def root():
    """根路径"""
    return {
        "name": "CityGuard API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "guard.server.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
