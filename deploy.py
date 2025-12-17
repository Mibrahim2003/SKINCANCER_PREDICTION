from app.workflow import skin_cancer_training_flow

if __name__ == "__main__":
    # Create a deployment that can be served locally
    # This keeps the process alive and listens for scheduled runs
    skin_cancer_training_flow.serve(
        name="skin-cancer-training-deployment",
        cron="0 0 * * 0",  # Run weekly on Sunday at midnight
        tags=["skin-cancer", "training"],
        description="Weekly training pipeline for Skin Cancer Detection model."
    )
