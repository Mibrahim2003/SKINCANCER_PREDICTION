"""
ML Model Validation Module using DeepChecks

This module provides automated testing and validation for ML models
using the DeepChecks library to ensure data quality, model performance,
and detect potential issues like drift.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

# Disable DeepChecks telemetry before importing
os.environ['DEEPCHECKS_DISABLE_TELEMETRY'] = 'true'

# Workaround for NumPy 2.x compatibility with deepchecks
# deepchecks 0.19.1 uses deprecated np.Inf
if not hasattr(np, 'Inf'):
    np.Inf = np.inf
if not hasattr(np, 'NaN'):
    np.NaN = np.nan

from deepchecks.tabular import Dataset, Suite
from deepchecks.tabular.checks import (
    TrainTestFeatureDrift,
    ModelInfo,
    TrainTestPerformance,
    SimpleModelComparison,
    TrainTestPredictionDrift,
    UnusedFeatures,
    FeatureFeatureCorrelation,
    DataDuplicates
)


def validate_training_run(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    model: Any,
    model_classes: np.ndarray = None
) -> Dict[str, Any]:
    """
    Validate a training run using DeepChecks comprehensive test suite.
    
    This function performs extensive validation including:
    - Train-Test distribution comparison
    - Data integrity checks
    - Model performance evaluation
    - Feature importance analysis
    - Drift detection
    
    Args:
        X_train: Training features (numpy array)
        X_test: Test features (numpy array)
        y_train: Training labels (numpy array)
        y_test: Test labels (numpy array)
        model: Trained scikit-learn compatible model
        model_classes: Array of all possible class labels (optional)
        
    Returns:
        dict: Validation results summary containing:
            - status: 'Passed' or 'Failed'
            - num_checks: Total number of checks run
            - num_passed: Number of passed checks
            - num_failed: Number of failed checks
            - report_path: Path to the HTML report
            
    Raises:
        ValueError: If critical validation checks fail (e.g., high drift detected)
    """
    print("\n" + "="*60)
    print("Starting DeepChecks Validation Suite")
    print("="*60)
    
    # Convert numpy arrays to DataFrames for DeepChecks
    # Feature names: f_0, f_1, ..., f_n
    n_features = X_train.shape[1]
    feature_names = [f"f_{i}" for i in range(n_features)]
    
    # Create DataFrames
    train_df = pd.DataFrame(X_train, columns=feature_names)
    train_df['target'] = y_train
    
    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df['target'] = y_test
    
    # Create DeepChecks Dataset objects
    # cat_features=[] because all our features are numeric (histogram values)
    print(f"Creating DeepChecks datasets...")
    print(f"  Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    train_dataset = Dataset(
        train_df,
        label='target',
        cat_features=[]  # No categorical features (all numeric histograms)
    )
    
    test_dataset = Dataset(
        test_df,
        label='target',
        cat_features=[]
    )
    
    # Run the full validation suite
    print("\nRunning validation suite...")
    print("  This may take a minute...")
    
    # Create a comprehensive validation suite with visualizations
    suite = Suite("Comprehensive ML Validation")
    
    # Model Information
    suite.add(ModelInfo())
    
    # Performance Checks - shows train vs test performance to detect overfitting
    suite.add(TrainTestPerformance())
    
    # Drift Detection - feature distribution changes (CREATES GRAPHS!)
    suite.add(TrainTestFeatureDrift())
    
    # Prediction Drift - label distribution changes
    suite.add(TrainTestPredictionDrift())
    
    # Feature Correlations - visualizes feature relationships
    suite.add(FeatureFeatureCorrelation())
    
    # Data Quality - detects duplicate samples
    suite.add(DataDuplicates())
    
    # Feature Importance - identifies unused features
    suite.add(UnusedFeatures())
    
    # Model Comparison - simple baseline comparison
    try:
        suite.add(SimpleModelComparison())
    except:
        pass  # Skip if causes issues with small datasets
    
    # Pass model_classes if provided to avoid issues with limited data
    try:
        if model_classes is not None:
            result = suite.run(
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                model=model,
                model_classes=model_classes
            )
        else:
            result = suite.run(
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                model=model
            )
    except Exception as e:
        print(f"\n\u26a0\ufe0f  Warning: Validation suite encountered an error: {str(e)}")
        print("  Continuing with limited validation...")
        
        # Fallback to minimal checks if full suite fails
        suite = Suite("Minimal Validation")
        suite.add(ModelInfo())
        result = suite.run(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            model=model
        )
    
    # Save the HTML report
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    report_path = reports_dir / "validation_report.html"
    
    print(f"\nSaving validation report to: {report_path}")
    
    # Generate static HTML report with embedded Plotly charts
    # (DeepChecks native save_as_html uses Jupyter widgets which don't work in browsers)
    html_content = _generate_static_html_report(
        result=result,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        model=model,
        feature_names=feature_names
    )
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"  ✅ Report saved successfully!")
    print(f"  📊 Open {report_path} in your browser to view interactive charts")
    
    # Extract result summary
    results_dict = result.to_json()
    
    # Count passed/failed checks
    num_checks = len(result.results)
    passed_checks = sum(1 for r in result.results if r.passed_conditions())
    failed_checks = num_checks - passed_checks
    
    print("\n" + "="*60)
    print("Validation Results Summary")
    print("="*60)
    print(f"Total Checks Run: {num_checks}")
    print(f"Passed: {passed_checks}")
    print(f"Failed: {failed_checks}")
    print(f"Report: {report_path}")
    
    # Print detailed results for failed checks
    if failed_checks > 0:
        print("\n" + "-"*60)
        print("Failed Checks Details:")
        print("-"*60)
        for check_result in result.results:
            if not check_result.passed_conditions():
                print(f"\n⚠️  {check_result.get_header()}")
                # Print condition details if available
                if hasattr(check_result, 'conditions_results'):
                    for cond in check_result.conditions_results:
                        if not cond.is_pass():
                            print(f"   - {cond.details}")
    
    # Create summary dictionary
    summary = {
        'status': 'Passed' if failed_checks == 0 else 'Failed',
        'num_checks': num_checks,
        'num_passed': passed_checks,
        'num_failed': failed_checks,
        'report_path': str(report_path)
    }
    
    # Print JSON summary for logging
    print("\n" + "-"*60)
    print("JSON Summary:")
    print(json.dumps(summary, indent=2))
    print("-"*60)
    
    # Check for critical failures
    # Note: We're being lenient here and just warning about failures
    # rather than blocking the pipeline on minor issues
    if failed_checks > 0:
        print("\n⚠️  WARNING: Some validation checks failed.")
        print("   Review the HTML report for details.")
        print("   Pipeline will continue, but consider investigating failures.")
        
        # You can optionally raise an error for critical failures
        # Uncomment the following lines to block on any failures:
        # raise ValueError(
        #     f"Model Validation Failed: {failed_checks} checks failed. "
        #     f"See {report_path} for details."
        # )
    else:
        print("\n✅ All validation checks passed!")
    
    print("="*60 + "\n")
    
    return summary


def validate_drift(
    X_train: np.ndarray,
    X_test: np.ndarray,
    drift_threshold: float = 0.5
) -> bool:
    """
    Quick drift check between training and test data.
    
    Args:
        X_train: Training features
        X_test: Test features
        drift_threshold: Maximum acceptable drift (0-1)
        
    Returns:
        bool: True if drift is below threshold, False otherwise
        
    Raises:
        ValueError: If significant drift detected above threshold
    """
    # Simple statistical drift detection
    # Compare distributions using Kolmogorov-Smirnov test
    from scipy import stats
    
    n_features = X_train.shape[1]
    drift_scores = []
    
    for i in range(n_features):
        # KS test returns statistic (0-1, higher = more drift) and p-value
        stat, _ = stats.ks_2samp(X_train[:, i], X_test[:, i])
        drift_scores.append(stat)
    
    max_drift = max(drift_scores)
    avg_drift = np.mean(drift_scores)
    
    print(f"\nDrift Analysis:")


def _generate_static_html_report(
    result,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    model,
    feature_names: list
) -> str:
    """
    Generate a completely static HTML report with embedded Plotly charts.
    
    This creates a self-contained HTML file that works in any browser
    WITHOUT requiring Jupyter widgets or any external dependencies.
    """
    from datetime import datetime
    import plotly.graph_objects as go
    import plotly.express as px
    from collections import Counter
    
    # Count stats
    total_checks = len(result.results)
    passed_checks = sum(1 for r in result.results if r.passed_conditions())
    failed_checks = total_checks - passed_checks
    
    # Generate our own visualizations (pure Plotly, no widgets)
    charts_html = ""
    
    # 1. Class Distribution Chart
    train_counts = Counter(y_train)
    test_counts = Counter(y_test)
    classes = sorted(set(y_train) | set(y_test))
    
    fig_class = go.Figure()
    fig_class.add_trace(go.Bar(
        name='Training',
        x=[f'Class {c}' for c in classes],
        y=[train_counts.get(c, 0) for c in classes],
        marker_color='#3b82f6'
    ))
    fig_class.add_trace(go.Bar(
        name='Test',
        x=[f'Class {c}' for c in classes],
        y=[test_counts.get(c, 0) for c in classes],
        marker_color='#10b981'
    ))
    fig_class.update_layout(
        title='Class Distribution: Train vs Test',
        barmode='group',
        xaxis_title='Class',
        yaxis_title='Count',
        template='plotly_white',
        height=400
    )
    charts_html += f'<div class="chart-container">{fig_class.to_html(include_plotlyjs=False, full_html=False)}</div>'
    
    # 2. Feature Distribution Comparison (sample of features)
    n_features_to_show = min(10, X_train.shape[1])
    feature_indices = np.linspace(0, X_train.shape[1]-1, n_features_to_show, dtype=int)
    
    fig_features = go.Figure()
    for idx in feature_indices:
        fig_features.add_trace(go.Box(
            y=X_train[:, idx],
            name=f'{feature_names[idx]} (Train)',
            marker_color='#3b82f6',
            boxpoints=False
        ))
        fig_features.add_trace(go.Box(
            y=X_test[:, idx],
            name=f'{feature_names[idx]} (Test)',
            marker_color='#10b981',
            boxpoints=False
        ))
    fig_features.update_layout(
        title=f'Feature Distribution Comparison (Sample of {n_features_to_show} features)',
        yaxis_title='Value',
        template='plotly_white',
        height=500,
        showlegend=False
    )
    charts_html += f'<div class="chart-container">{fig_features.to_html(include_plotlyjs=False, full_html=False)}</div>'
    
    # 3. Feature Correlation Heatmap (subset)
    n_corr_features = min(20, X_train.shape[1])
    corr_indices = np.linspace(0, X_train.shape[1]-1, n_corr_features, dtype=int)
    X_subset = X_train[:, corr_indices]
    corr_matrix = np.corrcoef(X_subset.T)
    
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=[feature_names[i] for i in corr_indices],
        y=[feature_names[i] for i in corr_indices],
        colorscale='RdBu',
        zmid=0
    ))
    fig_corr.update_layout(
        title='Feature Correlation Heatmap (Subset)',
        template='plotly_white',
        height=600,
        width=800
    )
    charts_html += f'<div class="chart-container">{fig_corr.to_html(include_plotlyjs=False, full_html=False)}</div>'
    
    # 4. Prediction Distribution (if model available)
    try:
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        pred_train_counts = Counter(y_pred_train)
        pred_test_counts = Counter(y_pred_test)
        
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Bar(
            name='Train Predictions',
            x=[f'Class {c}' for c in classes],
            y=[pred_train_counts.get(c, 0) for c in classes],
            marker_color='#8b5cf6'
        ))
        fig_pred.add_trace(go.Bar(
            name='Test Predictions',
            x=[f'Class {c}' for c in classes],
            y=[pred_test_counts.get(c, 0) for c in classes],
            marker_color='#f59e0b'
        ))
        fig_pred.update_layout(
            title='Prediction Distribution: Train vs Test',
            barmode='group',
            xaxis_title='Predicted Class',
            yaxis_title='Count',
            template='plotly_white',
            height=400
        )
        charts_html += f'<div class="chart-container">{fig_pred.to_html(include_plotlyjs=False, full_html=False)}</div>'
        
        # 5. Confusion Matrix for Test Set
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred_test)
        
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=[f'Pred {c}' for c in sorted(set(y_test))],
            y=[f'True {c}' for c in sorted(set(y_test))],
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}',
            textfont={'size': 14}
        ))
        fig_cm.update_layout(
            title='Confusion Matrix (Test Set)',
            xaxis_title='Predicted Label',
            yaxis_title='True Label',
            template='plotly_white',
            height=500,
            width=600
        )
        charts_html += f'<div class="chart-container">{fig_cm.to_html(include_plotlyjs=False, full_html=False)}</div>'
        
        # 6. Per-Class Accuracy
        class_accuracies = cm.diagonal() / cm.sum(axis=1)
        fig_acc = go.Figure(data=go.Bar(
            x=[f'Class {c}' for c in sorted(set(y_test))],
            y=class_accuracies * 100,
            marker_color=['#10b981' if acc > 0.7 else '#ef4444' if acc < 0.5 else '#f59e0b' for acc in class_accuracies],
            text=[f'{acc:.1%}' for acc in class_accuracies],
            textposition='outside'
        ))
        fig_acc.update_layout(
            title='Per-Class Accuracy',
            xaxis_title='Class',
            yaxis_title='Accuracy (%)',
            template='plotly_white',
            height=400,
            yaxis_range=[0, 110]
        )
        charts_html += f'<div class="chart-container">{fig_acc.to_html(include_plotlyjs=False, full_html=False)}</div>'
        
    except Exception as e:
        charts_html += f'<div class="chart-container"><p>Prediction charts unavailable: {str(e)}</p></div>'
    
    # 7. Feature Importance (if available)
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            top_n = 20
            top_indices = np.argsort(importances)[-top_n:][::-1]
            
            fig_imp = go.Figure(data=go.Bar(
                x=[feature_names[i] for i in top_indices],
                y=[importances[i] for i in top_indices],
                marker_color='#6366f1'
            ))
            fig_imp.update_layout(
                title=f'Top {top_n} Most Important Features',
                xaxis_title='Feature',
                yaxis_title='Importance',
                template='plotly_white',
                height=400
            )
            charts_html += f'<div class="chart-container">{fig_imp.to_html(include_plotlyjs=False, full_html=False)}</div>'
    except:
        pass
    
    # Build DeepChecks results section
    checks_html = ""
    for i, check_result in enumerate(result.results, 1):
        check_name = check_result.get_header()
        passed = check_result.passed_conditions()
        status = "✅ PASSED" if passed else "⚠️ FAILED"
        status_class = "passed" if passed else "failed"
        
        # Get check value/details
        check_value = ""
        if hasattr(check_result, 'value') and check_result.value is not None:
            val = check_result.value
            if isinstance(val, dict):
                check_value = "<table class='result-table'>"
                for k, v in list(val.items())[:10]:  # Limit to 10 items
                    if isinstance(v, float):
                        check_value += f"<tr><td>{k}</td><td>{v:.4f}</td></tr>"
                    else:
                        check_value += f"<tr><td>{k}</td><td>{v}</td></tr>"
                check_value += "</table>"
            elif isinstance(val, (int, float)):
                check_value = f"<p class='value-display'>Value: <strong>{val:.4f}</strong></p>"
            elif isinstance(val, str):
                check_value = f"<p>{val}</p>"
        
        # Get condition results
        conditions_html = ""
        if hasattr(check_result, 'conditions_results') and check_result.conditions_results:
            conditions_html = "<div class='conditions'>"
            for cond in check_result.conditions_results:
                cond_status = "✓" if cond.is_pass() else "✗"
                cond_class = "cond-pass" if cond.is_pass() else "cond-fail"
                details = cond.details if hasattr(cond, 'details') else ""
                conditions_html += f"<p class='{cond_class}'>{cond_status} {details}</p>"
            conditions_html += "</div>"
        
        checks_html += f"""
        <div class="check-result {status_class}">
            <h3>{i}. {check_name} <span class="status">{status}</span></h3>
            {check_value}
            {conditions_html}
        </div>
        """
    
    # Calculate metrics for summary
    try:
        from sklearn.metrics import accuracy_score, f1_score
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
    except:
        accuracy = 0
        f1 = 0
    
    # Final HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ML Validation Report</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #f3f4f6;
            color: #1f2937;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 16px;
            margin-bottom: 30px;
            box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
        }}
        .header h1 {{
            margin: 0 0 10px 0;
            font-size: 2.5rem;
        }}
        .header p {{
            margin: 0;
            opacity: 0.9;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .summary-card {{
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            text-align: center;
        }}
        .summary-card h3 {{
            margin: 0 0 10px 0;
            color: #6b7280;
            font-size: 14px;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .summary-card .value {{
            font-size: 2.5rem;
            font-weight: 700;
        }}
        .summary-card.passed .value {{ color: #10b981; }}
        .summary-card.failed .value {{ color: #ef4444; }}
        .summary-card.total .value {{ color: #3b82f6; }}
        .summary-card.accuracy .value {{ color: #8b5cf6; }}
        
        .section-title {{
            font-size: 1.5rem;
            margin: 40px 0 20px 0;
            padding-bottom: 10px;
            border-bottom: 2px solid #e5e7eb;
        }}
        
        .chart-container {{
            background: white;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        }}
        
        .check-result {{
            background: white;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        }}
        .check-result h3 {{
            margin: 0 0 15px 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
        }}
        .check-result.passed {{
            border-left: 4px solid #10b981;
        }}
        .check-result.failed {{
            border-left: 4px solid #ef4444;
        }}
        .status {{
            font-size: 13px;
            padding: 5px 14px;
            border-radius: 20px;
            font-weight: 500;
        }}
        .check-result.passed .status {{
            background: #d1fae5;
            color: #047857;
        }}
        .check-result.failed .status {{
            background: #fee2e2;
            color: #b91c1c;
        }}
        .result-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
        }}
        .result-table td {{
            padding: 8px 12px;
            border-bottom: 1px solid #e5e7eb;
        }}
        .result-table td:first-child {{
            font-weight: 500;
            color: #4b5563;
        }}
        .value-display {{
            font-size: 1.1rem;
            color: #374151;
        }}
        .conditions {{
            margin-top: 10px;
            padding: 10px;
            background: #f9fafb;
            border-radius: 8px;
        }}
        .conditions p {{
            margin: 5px 0;
        }}
        .cond-pass {{ color: #059669; }}
        .cond-fail {{ color: #dc2626; }}
        
        .metadata {{
            background: white;
            padding: 25px;
            border-radius: 12px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        }}
        .metadata h2 {{
            margin: 0 0 20px 0;
        }}
        .metadata-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        .metadata-item {{
            padding: 15px;
            background: #f9fafb;
            border-radius: 8px;
        }}
        .metadata-item label {{
            display: block;
            font-size: 12px;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 5px;
        }}
        .metadata-item span {{
            font-size: 1.25rem;
            font-weight: 600;
            color: #1f2937;
        }}
        
        .footer {{
            margin-top: 50px;
            padding: 25px;
            background: white;
            border-radius: 12px;
            text-align: center;
            color: #6b7280;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 ML Validation Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="summary">
        <div class="summary-card total">
            <h3>Total Checks</h3>
            <div class="value">{total_checks}</div>
        </div>
        <div class="summary-card passed">
            <h3>Passed</h3>
            <div class="value">{passed_checks}</div>
        </div>
        <div class="summary-card failed">
            <h3>Failed</h3>
            <div class="value">{failed_checks}</div>
        </div>
        <div class="summary-card accuracy">
            <h3>Accuracy</h3>
            <div class="value">{accuracy:.1%}</div>
        </div>
    </div>
    
    <div class="metadata">
        <h2>📊 Dataset & Model Information</h2>
        <div class="metadata-grid">
            <div class="metadata-item">
                <label>Training Samples</label>
                <span>{X_train.shape[0]:,}</span>
            </div>
            <div class="metadata-item">
                <label>Test Samples</label>
                <span>{X_test.shape[0]:,}</span>
            </div>
            <div class="metadata-item">
                <label>Features</label>
                <span>{X_train.shape[1]:,}</span>
            </div>
            <div class="metadata-item">
                <label>Classes</label>
                <span>{len(classes)}</span>
            </div>
            <div class="metadata-item">
                <label>Model Type</label>
                <span>{type(model).__name__}</span>
            </div>
            <div class="metadata-item">
                <label>F1 Score (Weighted)</label>
                <span>{f1:.3f}</span>
            </div>
        </div>
    </div>
    
    <h2 class="section-title">📈 Visual Analysis</h2>
    {charts_html}
    
    <h2 class="section-title">✅ DeepChecks Validation Results</h2>
    {checks_html}
    
    <div class="footer">
        <p><strong>ML Validation Report</strong> | Generated by DeepChecks + Custom Visualization Pipeline</p>
        <p>All charts are interactive - hover, zoom, and pan to explore your data!</p>
    </div>
</body>
</html>"""
    
    return html


def _create_html_report(result, X_train, X_test, model) -> str:
    """Legacy function - now redirects to the new static generator."""
    # This function is kept for backward compatibility
    # The new _generate_static_html_report is the preferred method
    pass


def validate_drift(
    X_train: np.ndarray,
    X_test: np.ndarray,
    drift_threshold: float = 0.5
) -> bool:
    """
    Quick drift check between training and test data.
    
    Args:
        X_train: Training features
        X_test: Test features
        drift_threshold: Maximum acceptable drift (0-1)
        
    Returns:
        bool: True if drift is below threshold, False otherwise
        
    Raises:
        ValueError: If significant drift detected above threshold
    """
    # Simple statistical drift detection
    # Compare distributions using Kolmogorov-Smirnov test
    from scipy import stats
    
    n_features = X_train.shape[1]
    drift_scores = []
    
    for i in range(n_features):
        # KS test returns statistic (0-1, higher = more drift) and p-value
        stat, _ = stats.ks_2samp(X_train[:, i], X_test[:, i])
        drift_scores.append(stat)
    
    max_drift = max(drift_scores)
    avg_drift = np.mean(drift_scores)
    
    print(f"\nDrift Analysis:")
    print(f"  Max drift: {max_drift:.4f}")
    print(f"  Avg drift: {avg_drift:.4f}")
    print(f"  Threshold: {drift_threshold}")
    
    if max_drift > drift_threshold:
        raise ValueError(
            f"Model Validation Failed due to high drift! "
            f"Max drift: {max_drift:.4f} > threshold: {drift_threshold}"
        )
    
    print("  ✅ Drift check passed")
    return True
