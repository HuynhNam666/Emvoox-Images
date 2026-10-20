"""Đăng nhập tối thiểu cho server: một token dùng chung (API_TOKEN). Chưa phải hệ phân quyền nhiều người dùng.

Token được chấp nhận qua header `Authorization: Bearer`, `X-API-Token` hoặc cookie HttpOnly `emvoox_token` (cookie để thẻ <img>/<video>
của giao diện web tải được tài sản). API_TOKEN rỗng => tắt xác thực (chỉ dùng localhost). Luôn đặt HTTPS phía trước khi ra Internet.
"""
import hmac

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.core.config import settings

OPEN_PATHS = {"/api/v1/health/live", "/api/v1/health/ready", "/api/v1/auth/login", "/api/v1/auth/status", "/api/v1/auth/logout"}
router = APIRouter(prefix="/api/v1/auth")


def _token_from(request: Request) -> str:
    h = request.headers.get("authorization", "")
    if h.lower().startswith("bearer "):
        return h[7:].strip()
    return request.headers.get("x-api-token") or request.cookies.get("emvoox_token") or ""


def authorized(request: Request) -> bool:
    return (not settings.api_token) or hmac.compare_digest(_token_from(request).encode(), settings.api_token.encode())


class TokenAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        p = request.url.path
        if settings.api_token and p.startswith("/api/") and p not in OPEN_PATHS and request.method != "OPTIONS" and not authorized(request):
            return JSONResponse({"detail": "cần đăng nhập (API_TOKEN)"}, status_code=401)
        return await call_next(request)


class Login(BaseModel):
    token: str


@router.get("/status")
def status(request: Request):
    return {"required": bool(settings.api_token), "authenticated": authorized(request)}


@router.post("/login")
def login(body: Login, response: Response):
    if settings.api_token and not hmac.compare_digest(body.token.encode(), settings.api_token.encode()):
        return JSONResponse({"detail": "token sai"}, status_code=401)
    r = JSONResponse({"ok": True})
    if settings.api_token:
        r.set_cookie("emvoox_token", body.token, httponly=True, samesite="strict", secure=settings.cookie_secure, max_age=30 * 86400)
    return r


@router.post("/logout")
def logout():
    r = JSONResponse({"ok": True})
    r.delete_cookie("emvoox_token")
    return r
