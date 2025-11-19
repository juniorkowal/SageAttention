import triton
import functools
import os


AUTOTUNE = os.environ.get('SAGE_ATTENTION_TRITON_AMD_AUTOTUNE', '0').lower() in ('1', 'true', 'yes')


def get_cdna_autotune_configs():
    return [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'waves_per_eu': 2}, num_stages=1,
                      num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 2}, num_stages=1,
                      num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 3}, num_stages=1,
                      num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 1}, num_stages=1,
                      num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'waves_per_eu': 2}, num_stages=1,
                      num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'waves_per_eu': 1}, num_stages=1,
                      num_warps=4),
        # Fall-back config.
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'waves_per_eu': 1}, num_stages=1,
                      num_warps=4),
    ]


def get_rdna_autotune_configs():
    return [
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'waves_per_eu': 4}, num_stages=1,
                      num_warps=2),
        # triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'waves_per_eu': 2}, num_stages=1,
        #               num_warps=2),
        # triton.Config({'BLOCK_M': 32, 'BLOCK_N': 16, 'waves_per_eu': 4}, num_stages=1,
        #               num_warps=2),
        # triton.Config({'BLOCK_M': 32, 'BLOCK_N': 16, 'waves_per_eu': 2}, num_stages=1,
        #               num_warps=2),
        # triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'waves_per_eu': 4}, num_stages=1,
        #               num_warps=2),
        # triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'waves_per_eu': 2}, num_stages=1,
        #               num_warps=2),
        # # Fall-back config.
        # triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'waves_per_eu': 1}, num_stages=1,
        #               num_warps=2),
    ]

# -------------------------------
# Runtime info
# -------------------------------
@functools.cache
def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"

@functools.cache
def get_arch():
    return triton.runtime.driver.active.get_current_target().arch

@functools.cache
def is_cdna():
    return is_hip() and get_arch() in ('gfx908', 'gfx90a', 'gfx940', 'gfx941', 'gfx942', 'gfx950')

@functools.cache
def is_rdna():
    return is_hip() and get_arch() in ("gfx1030", "gfx1100", "gfx1101", "gfx1102", "gfx1200", "gfx1201")

def get_autotune_configs():
    if AUTOTUNE:
        if is_rdna():
            return get_rdna_autotune_configs()
        elif is_cdna():
            return get_cdna_autotune_configs()
        else:
            raise ValueError("Unknown Device Type")
    else:
        return [
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 64, "waves_per_eu": 1},
                num_stages=1,
                num_warps=4,
            ),
        ]


autotune_configs = get_autotune_configs()




# @functools.cache
# def arch_supports_fp8():
#     return is_hip() and get_arch() in ('gfx942')