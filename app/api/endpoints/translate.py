"""
POST /api — single TranslateKH-compatible translation endpoint.

Clients switching from the real TranslateKH API need only change:
  - base URL  (https://translatekh.com → http://your-host)
  - credentials  (their TranslateKH key → API_USERNAME / API_PASSWORD)

Request and response schemas are identical.
"""
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.dependencies import get_translation_service, verify_credentials
from app.schemas.translate import TranslateRequest, TranslateResponse
from app.services.translation import TranslationService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["translation"])


@router.post(
    "/api",
    response_model=TranslateResponse,
    summary="Translate text (TranslateKH-compatible)",
    description=(
        "Translate one or more texts in a single request.\n\n"
        "**Schema** matches the TranslateKH API exactly — "
        "`input_text` is always an array, `translate_text` is always an array "
        "of the same length in the same order.\n\n"
        "**Provider fallback** (transparent to the client):\n"
        "1. Google Cloud Translation — deterministic NMT\n"
        "2. Gemini Flash Lite — LLM fallback\n"
        "3. OpenRouter free — last resort\n\n"
        "**Authentication**: HTTP Basic Auth (`-u username:password`).\n\n"
        "**Caching**: identical requests are served from Redis (24 h TTL)."
    ),
    responses={
        401: {"description": "Invalid or missing credentials"},
        429: {"description": "Rate limit exceeded"},
        502: {"description": "All translation providers failed"},
    },
)
async def translate(
    body: TranslateRequest,
    service: Annotated[TranslationService, Depends(get_translation_service)],
    _auth: Annotated[None, Depends(verify_credentials)],
) -> TranslateResponse:
    try:
        results = await service.translate(
            texts=body.input_text,
            src_lang=body.src_lang,
            tgt_lang=body.tgt_lang,
        )
    except NotImplementedError as exc:
        logger.warning("translate: unsupported operation: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="This translation operation is not supported.",
        ) from exc
    except Exception as exc:
        logger.error("translate: provider failure: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Translation failed. Please try again later.",
        ) from exc

    return TranslateResponse(translate_text=results)
