# Profiling DFlash.

Refer original [DFlash](https://github.com/z-lab/dflash)

# Reproduce 

Take Qwen3-4B as example.

#### ENV

```
git clone https://github.com/z-lab/dflash
python -m venv env_dflash
cd dflash
source ../env_dflash/bin/activate
uv pip install -e .
```

#### Run sample

```
./run.sh
```