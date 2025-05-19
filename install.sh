python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install magic pdf model weights
pip install huggingface_hub
wget https://github.com/opendatalab/MinerU/raw/master/scripts/download_models_hf.py -O download_models_hf.py
python download_models_hf.py

