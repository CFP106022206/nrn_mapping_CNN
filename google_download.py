# %%
import gdown

url = "https://drive.google.com/drive/folders/1qJNtP1vEdzSuWqs1eEXvRUSnConRuINU?usp=share_link"
output = "/Users/richard/Downloads/D5_no_file/test.swc"
# gdown.download(url, output)
gdown.download_folder(url, quiet=True)
# %%
