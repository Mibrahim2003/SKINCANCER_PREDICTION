#!/usr/bin/env python
"""
Simple runner script for the ML pipeline
Usage: python run_pipeline.py [sample_limit]
"""

import sys
import os

# Ensure we're in the project root
project_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_root)
sys.path.insert(0, project_root)

# Import and run the workflow
from app.workflow import skin_cancer_training_flow

if __name__ == "__main__":
    # Parse command line arguments
    limit = 500  # Default sample limit
    if len(sys.argv) > 1:
        try:
            limit = int(sys.argv[1])
        except ValueError:
            print(f"Invalid limit: {sys.argv[1]}. Using default: 500")
    
    print(f"\n{'='*60}")
    print(f"🚀 Starting Skin Cancer Model Training Pipeline")
    print(f"📊 Sample Limit: {limit}")
    print(f"{'='*60}\n")
    
    # Run the flow
    skin_cancer_training_flow(limit=limit)
    
    print(f"\n{'='*60}")
    print(f"✅ Pipeline Complete!")
    print(f"📊 View report: reports/validation_report.html")
    print(f"{'='*60}\n")
