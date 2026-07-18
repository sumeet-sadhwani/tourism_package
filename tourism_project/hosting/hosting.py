from huggingface_hub import HfApi
import os


token=os.getenv("HF_TOKEN")   # please use your token

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="tourism_project/deployment",     # the local folder containing your files
    # replace with your repoid
    repo_id="SumeetSadhwani/TourismPackage",          # the target repo

    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
