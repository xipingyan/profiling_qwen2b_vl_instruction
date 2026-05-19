from pathlib import Path
import sys

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset, load_from_disk

# Verify whisper model.
# pyenv:
# uv pip install transformers datasets torch torchcodec

# convert to openvino script:
# ================================================================
# model_id='openai/whisper-tiny'
# "$OPTIMUM_CLI" export openvino \
#   --model "$model_id" \
#   --task automatic-speech-recognition \
#   "$model_id/FP16"


script_dir = Path(__file__).resolve().parent
model_dir = (script_dir / "../../models/openai/whisper-tiny/").resolve()
model_id = str(model_dir) if model_dir.exists() else "openai/whisper-tiny"

# load model and processor
processor = WhisperProcessor.from_pretrained(model_id)
model = WhisperForConditionalGeneration.from_pretrained(model_id)
model.config.forced_decoder_ids = None

# load dummy dataset from local first
# 1) If you already have a `save_to_disk` folder, place it here:
local_saved_ds_dir = script_dir / "local_datasets" / "librispeech_asr_dummy_clean_validation"
# 2) If you downloaded dataset repo files from webpage/snapshot, place them here:
local_repo_ds_dir = script_dir / "hf-internal-testing" / "librispeech_asr_dummy"

if local_saved_ds_dir.exists() and (local_saved_ds_dir / "dataset_info.json").exists():
	ds = load_from_disk(str(local_saved_ds_dir))
elif local_repo_ds_dir.exists():
	try:
		ds = load_dataset(str(local_repo_ds_dir), "clean", split="validation")
	except ValueError:
		ds = load_dataset(str(local_repo_ds_dir), split="validation")
	local_saved_ds_dir.parent.mkdir(parents=True, exist_ok=True)
	ds.save_to_disk(str(local_saved_ds_dir))
else:
	print("Local dataset not found.")
	print(f"Expected one of:\n  1) {local_saved_ds_dir}\n  2) {local_repo_ds_dir}")
	print("Please put your downloaded local dataset in one of these paths.")
	sys.exit(1)

sample = ds[0]["audio"]
input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features 

# generate token ids
predicted_ids = model.generate(input_features)
# decode token ids to text
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)
print("output (skip_special_tokens=False): ", transcription)

transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
print("output (skip_special_tokens=True): ", transcription)
