# 1. Library imports
import uvicorn
from fastapi import FastAPI
from lead_conv_base import Convert
import numpy as np
import pickle
import pandas as pd
# 2. Create the app object
app = FastAPI()
pickle_in = open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)

# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello Marketers'}

# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Bank Note with the confidence
@app.post('/predict')
def predict_conversion(data:Convert):
    data = data.dict()
    Region_Code=data['Region_Code']
    Reco_Insurance_Type=data['Reco_Insurance_Type']
    Upper_Age=data['Upper_Age']
    Reco_Policy_Cat=data['Reco_Policy_Cat']
   
    pred = classifier.predict_proba(np.array([[Region_Code,Reco_Insurance_Type,Upper_Age,Reco_Policy_Cat]]))[:,1]
    if(pred[0]>0.5):
        prediction="Customer will convert"+str(pred)
    else:
        prediction="Customer will not convert"+str(pred)
    return {
        'prediction': prediction
    }

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)