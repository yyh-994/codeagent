"""测试 Pydantic 请求/响应模型的参数校验"""
import pytest
from pydantic import ValidationError


class TestLoginRequest:
    def test_valid(self):
        from back.schemas import LoginRequest
        r = LoginRequest(username="admin", password="secret")
        assert r.username == "admin"
        assert r.password == "secret"

    def test_empty_username_fails(self):
        from back.schemas import LoginRequest
        with pytest.raises(ValidationError):
            LoginRequest(username="", password="secret")

    def test_empty_password_fails(self):
        from back.schemas import LoginRequest
        with pytest.raises(ValidationError):
            LoginRequest(username="admin", password="")

    def test_too_long_username(self):
        from back.schemas import LoginRequest
        with pytest.raises(ValidationError):
            LoginRequest(username="a" * 51, password="secret")


class TestTaskSubmitRequest:
    def test_valid_minimal(self):
        from back.schemas import TaskSubmitRequest
        r = TaskSubmitRequest(input={"messages": [{"role": "user", "content": "hello"}]})
        assert r.input == {"messages": [{"role": "user", "content": "hello"}]}
        assert r.config == {}

    def test_with_config(self):
        from back.schemas import TaskSubmitRequest
        r = TaskSubmitRequest(
            input={"messages": []},
            config={"configurable": {"thread_id": "th-123"}},
        )
        assert r.config["configurable"]["thread_id"] == "th-123"

    def test_missing_input_fails(self):
        from back.schemas import TaskSubmitRequest
        with pytest.raises(ValidationError):
            TaskSubmitRequest()


class TestTokenResponse:
    def test_structure(self):
        from back.schemas import TokenResponse
        r = TokenResponse(
            access_token="at-1",
            token_type="bearer",
            user_id="uid-1",
            role="vip",
            expires_in=1800,
        )
        assert r.access_token == "at-1"
        assert r.role == "vip"
        assert r.expires_in == 1800


class TestMessageResponse:
    def test_default(self):
        from back.schemas import MessageResponse
        r = MessageResponse()
        assert r.msg == "ok"
