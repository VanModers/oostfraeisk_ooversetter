from huggingface_hub import login, upload_folder

login()

upload_folder(
    folder_path="de_frs_model",
    repo_id="VanModers114/opus-mt-de-frs",
    repo_type="model",
    delete_patterns="*",
    ignore_patterns=[
        "__pycache__",
        "**/__pycache__/**",
        "*.pyc",
    ],
)