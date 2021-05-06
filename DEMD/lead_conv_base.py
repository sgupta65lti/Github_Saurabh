from pydantic import BaseModel
# 2. Class which describes Bank Notes measurements
class Convert(BaseModel):
    Region_Code: int 
    Reco_Insurance_Type: int 
    Upper_Age: int 
    Reco_Policy_Cat: int

