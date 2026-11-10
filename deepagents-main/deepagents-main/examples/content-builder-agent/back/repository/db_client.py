"""数据库客户端 — engine 构建 + 会话 + 初始化"""

import os

from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from back.common.config import DB_POOL_SIZE, DB_POOL_RECYCLE
from .entity import User, Role, Base

DATABASE_URL = (
    f"mysql+asyncmy://"
    f"{os.environ.get('MYSQL_USER', 'root')}:"
    f"{os.environ.get('MYSQL_PASSWORD', 'root')}@"
    f"{os.environ.get('MYSQL_HOST', 'localhost')}:"
    f"{os.environ.get('MYSQL_PORT', '3306')}/"
    f"{os.environ.get('MYSQL_DATABASE', 'content_builder_agent')}"
    f"?charset=utf8mb4"
)

# 创建异步连接池
engine = create_async_engine(
    DATABASE_URL,
    pool_size=DB_POOL_SIZE,
    max_overflow=10,
    pool_recycle=DB_POOL_RECYCLE,
    echo=False,
)

# 创建会话：绑定全局连接池，运行时申请一条 TCP 连接，执行 SQL
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def init_db():
    """建表 + 插入测试用户（如表为空）。"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with AsyncSessionLocal() as s:
        result = await s.execute(select(User))
        # 创建测试用户
        if len(result.scalars().all()) == 0:
            admin = User(username="admin")
            admin.set_password("admin123")
            admin.role_rel = Role(role="vip")
            s.add(admin)

            test = User(username="test")
            test.set_password("test123")
            test.role_rel = Role(role="basic")
            s.add(test)

            await s.commit()  # 发生异常，SQLAlchemy 自动回滚
