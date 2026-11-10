"""全局异常处理 — FastAPI app.exception_handler 注册"""

import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from back.common.exceptions import AppException, ErrorCode

logger = logging.getLogger("exception")


def register_exception_handler(app: FastAPI):
    @app.exception_handler(AppException)
    async def app_exception_handler(request: Request, exc: AppException):
        logger.warning("业务异常 | code=%s msg=%s detail=%s", exc.code, exc.message, exc.detail)
        return JSONResponse(status_code=exc.http_status, content=exc.to_response_dict())

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception("系统未知异常")
        return JSONResponse(status_code=500, content={
            "code": ErrorCode.INTERNAL_ERROR, "msg": "服务器内部错误", "data": {}
        })
