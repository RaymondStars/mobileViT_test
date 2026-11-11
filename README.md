# testing mobileViT performance with hugging face, this repo serves the following purpose:
1. Test and implement scripts for importing ImageNet-1k Dataset
2. Test and implement mobileViT with hugging face
3. Check the performance of smaller versions of the model (e.g. XXS)

# Prerequisites
1. Have conda
2. Have hugging face account and obtained an access token

# Steps
1. Run the following command in the terminal (replace YOUR_HUGGINGFACE_ACCESS_TOKEN with the actual token)
```
conda env create --file environment.yml
export HF_TOKEN="YOUR_HUGGINGFACE_ACCESS_TOKEN"
```
2. Run prepare_dataset.py to get the validation set
3. Run huggingface.py for validation, you may change the model name to different models for testing