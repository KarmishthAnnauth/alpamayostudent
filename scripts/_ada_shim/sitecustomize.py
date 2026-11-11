"""Cap this process's GPU memory fraction when A2E_MEM_FRAC is set."""
import os

_frac = os.environ.get("A2E_MEM_FRAC")
if _frac:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(float(_frac), 0)
            _t = torch.cuda.get_device_properties(0).total_memory / 2**30
            print(f"sitecustomize: capped at {float(_frac):.0%} of "
                  f"{_t:.1f} GiB = {float(_frac) * _t:.1f} GiB", flush=True)
    except Exception as e:                                          # noqa: BLE001
        print(f"sitecustomize: memory cap NOT applied ({e})", flush=True)
