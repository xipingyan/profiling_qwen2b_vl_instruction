source env_dflash/bin/activate

# Ensure latest local remote-code file is reloaded instead of stale HF cache copy.
rm -f ~/.cache/huggingface/modules/transformers_modules/dflash.py
rm -rf ~/.cache/huggingface/modules/transformers_modules/__pycache__

# EXPORT_OV=1 
python sample.py