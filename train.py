from src.preprocessing import preprocess_data
from src.model_training import train_model, save_model
from src.evaluation import evaluate_model

DATA_PATH = "dataset/train.csv"

# Step 1: preprocessing
X, y = preprocess_data(DATA_PATH)

# Step 2: training
model, X_test, y_test = train_model(X, y)

# Step 3: evaluation
evaluate_model(model, X_test, y_test)

# Step 4: save model
save_model(model)

print("Pipeline executed successfully")