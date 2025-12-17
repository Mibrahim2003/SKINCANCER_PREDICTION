"""Quick test to regenerate validation report with proper visualizations"""
import sys
sys.path.insert(0, 'c:\\Users\\ibrah\\Desktop\\New Project')

# Set up NumPy compatibility
import numpy as np
if not hasattr(np, 'Inf'):
    np.Inf = np.inf
if not hasattr(np, 'NaN'):
    np.NaN = np.nan

from app.ml_validation import _create_html_report
from deepchecks.tabular import Dataset, Suite
from deepchecks.tabular.checks import (
    TrainTestFeatureDrift,
    ModelInfo,
    TrainTestPerformance
)
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

print("Generating test validation report...")

# Create simple test data
X = np.random.rand(100, 10)
y = np.random.randint(0, 2, 100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train simple model
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, y_train)

# Create datasets
feature_names = [f"f_{i}" for i in range(X_train.shape[1])]
train_df = pd.DataFrame(X_train, columns=feature_names)
train_df['target'] = y_train

test_df = pd.DataFrame(X_test, columns=feature_names)
test_df['target'] = y_test

train_dataset = Dataset(train_df, label='target', cat_features=[])
test_dataset = Dataset(test_df, label='target', cat_features=[])

# Run validation suite
suite = Suite("Test Validation")
suite.add(ModelInfo())
suite.add(TrainTestPerformance())
suite.add(TrainTestFeatureDrift())

result = suite.run(
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    model=model
)

# Generate HTML
print("Creating HTML report...")
html = _create_html_report(result, X_train, X_test, model)

# Save
with open('reports/validation_report.html', 'w', encoding='utf-8') as f:
    f.write(html)

print("✅ Report saved to reports/validation_report.html")
print("Open it in your browser to see interactive charts!")
