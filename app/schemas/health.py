from pydantic import BaseModel


class ServiceStatus(BaseModel):
    status: str
    detail: str | None = None


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    services: dict[str, ServiceStatus]
