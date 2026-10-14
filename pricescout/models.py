"""Data model for scraped products, validated with pydantic."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field, field_validator


class Product(BaseModel):
    """One product as it appears on the target site.

    Validation guarantees that every exported row has a title, a URL and a
    non-negative price, so bad rows are caught at scrape time instead of in
    the client's spreadsheet.
    """

    url: str
    title: str
    price: float = Field(ge=0)
    currency: str = "GBP"
    rating: int | None = Field(default=None, ge=1, le=5)
    in_stock: bool = True
    stock_count: int | None = Field(default=None, ge=0)
    category: str | None = None
    upc: str | None = None
    description: str | None = None
    num_reviews: int | None = Field(default=None, ge=0)
    image_url: str | None = None
    scraped_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("title", "url")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = (value or "").strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator("category", "upc", "description")
    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @property
    def key(self) -> str:
        """Stable identifier used for de-duplication and price history."""
        return self.upc or self.url


# Column order used by every exporter, so CSV / Excel / Sheets always match.
EXPORT_COLUMNS = [
    "title",
    "price",
    "currency",
    "rating",
    "in_stock",
    "stock_count",
    "category",
    "upc",
    "num_reviews",
    "url",
    "image_url",
    "description",
    "scraped_at",
]
