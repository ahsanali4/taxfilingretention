from pydantic import BaseModel


class UserData(BaseModel):
    age: int
    income: float
    employment_type: str
    marital_status: str
    time_spent_on_platform: float
    number_of_sessions: int
    fields_filled_percentage: float
    previous_year_filing: int
    device_type: str
    referral_source: str
