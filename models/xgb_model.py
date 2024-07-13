import xgboost as xgb

def create_xgb_model():
    return xgb.XGBClassifier(eval_metric='merror', random_state=42)
