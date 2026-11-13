rom sklearn.metrics import mean_squared_error, r2_score
def evaluate_model(model, X_test, y_test):
 preds = model.predict(X_test)
 rmse = mean_squared_error(y_test, preds, squared=False)
 r2 = r2_score(y_test, preds)
 return {"RMSE": rmse, "R2": r2}