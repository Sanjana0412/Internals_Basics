import argparse
import joblib
import pandas as pd
import json

def predict():
    parser = argparse.ArgumentParser()
    parser.add_argument("--txn_amount", type=float)
    parser.add_argument("--merchant_risk_score", type=float)
    parser.add_argument("--is_international", type=int)
    parser.add_argument("--gateway_load", type=float)
    args = parser.parse_args()

    model = joblib.load("models/model.pkl")
    input_data = pd.DataFrame([[args.txn_amount, args.merchant_risk_score, args.is_international, args.gateway_load]], 
                               columns=['txn_amount', 'merchant_risk_score', 'is_international', 'gateway_load'])
    
    prediction = model.predict(input_data)[0]
    print(round(prediction, 4))

if __name__ == "__main__":
    predict()