from pydantic import BaseModel


class RewardScaleEntry(BaseModel):
    min_value: int
    max_value: int
    description: str
