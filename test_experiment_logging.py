"""Quick test to verify experiment logging functionality."""

from app.utils import log_experiment_result

# Test experiment logging with sample data
test_metrics = {
    "accuracy": 0.8542,
    "f1_score": 0.8321,
    "recall_melanoma": 0.7654
}

print("Testing experiment logging...")
log_experiment_result(
    model_name="RandomForest",
    params="n_estimators=200, max_depth=20, random_state=42",
    metrics=test_metrics
)
print("✅ Test complete! Check reports/experiments.csv")
