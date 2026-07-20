import logging
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware


logger = logging.getLogger(__name__)


class RequestTimeLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time

        if "classify" in request.url.path:
            logger.info(
                f"[{request.method}] {request.url.path} "
                f"| Статус: {response.status_code} "
                f"| Время инференса: {process_time:.2f} сек."
            )

        response.headers["X-Model-Process-Time"] = str(process_time)
        return response


def setup_middlewares(app: FastAPI, cors_origins: list[str]):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestTimeLoggingMiddleware)
