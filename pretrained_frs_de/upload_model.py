from huggingface_hub import login, upload_folder

login()

upload_folder(
    folder_path="pretrained_frs_de",
    repo_id="VanModers114/opus-mt-frs-de",
    repo_type="model",
    delete_patterns="*",
)
