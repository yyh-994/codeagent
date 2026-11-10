"""统一异常体系 — 所有层级共用"""

from typing import Optional, Any


# ── 错误码常量 ──
class ErrorCode:
    INTERNAL_ERROR = "INTERNAL_ERROR"  # 服务器未知内部错误兜底
    VALIDATION_ERROR = "VALIDATION_ERROR"
    SERVICE_ERROR = "SERVICE_ERROR"
    DB_ERROR = "DB_ERROR"
    EXTERNAL_SERVICE = "EXTERNAL_SERVICE"
    AUTH_ERROR = "AUTH_ERROR"
    TOKEN_EXPIRED = "TOKEN_EXPIRED"
    TOKEN_INVALID = "TOKEN_INVALID"
    LOGIN_FAILED = "LOGIN_FAILED"
    RATE_LIMITED = "RATE_LIMITED"


# ── 基类 ──
class AppException(Exception):
    def __init__(self, message: str, code: str = ErrorCode.INTERNAL_ERROR,
                 http_status: int = 500, detail: Optional[dict[str, Any]] = None):
        self.message = message
        self.code = code
        self.http_status = http_status
        self.detail = detail or {}
        super().__init__(message)

    def to_response_dict(self) -> dict:
        return {"code": self.code, "msg": self.message, "data": self.detail}


# ── 认证授权（401）──
class AuthException(AppException):
    def __init__(self, message: str = "认证失败", code: str = ErrorCode.AUTH_ERROR, detail: dict = None):
        super().__init__(message, code, http_status=401, detail=detail)

class LoginFailedException(AuthException):
    def __init__(self, message: str = "用户名或密码错误"):
        super().__init__(message, ErrorCode.LOGIN_FAILED)

class RateLimitException(AppException):
    def __init__(self, message: str = "请求过于频繁"):
        super().__init__(message, ErrorCode.RATE_LIMITED, http_status=429)


# ── 业务异常 ──
class NotFoundException(AppException):
    def __init__(self, message: str = "资源不存在"):
        super().__init__(message, ErrorCode.INPUT_ERROR, http_status=404)


class ValidationException(AppException):
    def __init__(self, message: str):
        super().__init__(message, ErrorCode.VALIDATION_ERROR, http_status=400)


class FileTooLargeException(AppException):
    def __init__(self, max_mb: int):
        super().__init__(
            f"文件不能超过 {max_mb}MB",
            ErrorCode.VALIDATION_ERROR,
            http_status=413,
        )


class ServiceException(AppException):
    def __init__(self, message: str, http_status: int = 422):
        super().__init__(message, ErrorCode.SERVICE_ERROR, http_status=http_status)


# ── 基础设施 ──
class DatabaseException(AppException):
    def __init__(self, message: str = "数据库异常"):
        super().__init__(message, ErrorCode.DB_ERROR, http_status=503)

class ExternalServiceException(AppException):
    def __init__(self, service_name: str, error_msg: str):
        super().__init__(f"外部服务 [{service_name}] 异常: {error_msg}", ErrorCode.EXTERNAL_SERVICE, http_status=502)
