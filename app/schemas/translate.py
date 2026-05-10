from pydantic import BaseModel, Field


class TranslateRequest(BaseModel):
    """
    TranslateKH-compatible request body.
    input_text is always an array — single sentences are wrapped in a one-element list.
    """
    input_text: list[str] = Field(
        ...,
        min_length=1,
        max_length=10,
        description=(
            "One or more texts to translate (1–10 items). "
            "Single sentences must still be sent as a one-element array."
        ),
    )
    src_lang: str = Field(..., description="Source language code, e.g. 'eng' or 'en'")
    tgt_lang: str = Field(..., description="Target language code, e.g. 'kh' or 'km'")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "input_text": ["Good morning!", "Welcome to Cambodia."],
                    "src_lang": "eng",
                    "tgt_lang": "kh",
                }
            ]
        }
    }


class TranslateResponse(BaseModel):
    """
    TranslateKH-compatible response body.
    translate_text mirrors input_text — same length, same order.
    """
    translate_text: list[str] = Field(
        ...,
        description=(
            "Translated texts in the same order as input_text. "
            "len(translate_text) == len(input_text) is always guaranteed."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "translate_text": [
                        "អរុណ​សួស្តី!",
                        "សូម​ស្វាគមន៍​មក​កម្ពុជា។",
                    ]
                }
            ]
        }
    }
