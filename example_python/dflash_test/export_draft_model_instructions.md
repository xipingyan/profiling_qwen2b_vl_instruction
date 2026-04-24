# Export DFlash Draft Model to OV::Model

## Work Path

```
/mnt/data_nvme1n1p1/xiping_workpath/mygithub/profiling_qwen2b_vl_instruction/example_python/dflash_test
```

## Latest Upgrade Summary

The conversion path has been upgraded to include `lm_head` in the exported OpenVINO graph.

Before:
- Exported graph output was draft hidden states only.

Now:
- Exported graph output is `target.lm_head(draft_hidden_states)` logits.
- This matches draft decode behavior where logits are produced after draft forward.

Reference for runtime logic:
`models/z-lab/Qwen3-4B-DFlash-b16/dflash.py` (see `draft_logits = target.lm_head(draft_result)`).

## How To Run Conversion (Export Only)

Use export-only mode to quickly generate OV files without running `spec_generate`.

```
cd /mnt/data_nvme1n1p1/xiping_workpath/mygithub/profiling_qwen2b_vl_instruction/example_python/dflash_test
source env_dflash/bin/activate
EXPORT_OV=1 EXPORT_ONLY=1 python sample.py
```

## Optional Environment Variables

- `EXPORT_OV=1`
	- Enable OV export.

- `EXPORT_ONLY=1`
	- Export and exit early.
	- Skip tokenizer load and `spec_generate`.

- `EXPORT_DRAFT_DIR=<dir>`
	- Output directory for exported model.
	- Default: `converted_draft_model`

Example:

```
EXPORT_OV=1 EXPORT_ONLY=1 EXPORT_DRAFT_DIR=converted_draft_model python sample.py
```

## Expected Output

After a successful run:

- XML: `converted_draft_model/draft_model.xml`
- BIN: `converted_draft_model/draft_model.bin`

Console should include:

```
== Start to export OV model.
== Start to save OV model.
== OV model exported and saved  successfully as converted_draft_model/draft_model.xml.
EXPORT_ONLY=1, skip spec_generate.
```

## Quick Sanity Check (lm_head in OV Graph)

```
grep -n "__module.lm_head/aten::linear/MatMul" converted_draft_model/draft_model.xml
```

If found, export graph includes `lm_head`.

