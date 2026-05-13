from huggingface_hub import login, upload_folder

login()

upload_folder(
    folder_path="nllb_frs_model",
    repo_id="VanModers114/East_Frisian_NLLB_Model",
    repo_type="model",
    delete_patterns="*",
    ignore_patterns=[
        "__pycache__",
        "**/__pycache__/**",
        "*.pyc",
    ],
)