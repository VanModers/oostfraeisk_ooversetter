from huggingface_hub import login, upload_folder

login()

upload_folder(
    folder_path="frs_de_model",
    repo_id="VanModers114/opus-mt-frs-de",
    repo_type="model",
    delete_patterns="*",
    ignore_patterns=[
        "__pycache__",
        "**/__pycache__/**",
        "*.pyc",
    ],
)
