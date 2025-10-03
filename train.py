from sklearn.tree import DecisionTreeRegressor 
from misc import load_data, preprocess_data, train_model, evaluate_model 
 
def run_decision_tree_workflow(): 
    print("--- Starting Decision Tree Model Workflow ---") 
    df = load_data() 
    X_train, X_test, y_train, y_test = preprocess_data(df) 
 
    dt_model = DecisionTreeRegressor(random_state=42) 
    trained_model = train_model(dt_model, X_train, y_train) 
 
    mse = evaluate_model(trained_model, X_test, y_test) 
 
    print(f"\n\x1B[32m\u2705 DecisionTreeRegressor Average MSE: {mse:.4f}\x1B[0m") 
 
if __name__ == "__main__": 
    run_decision_tree_workflow() 
