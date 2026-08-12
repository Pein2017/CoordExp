"""Compact, non-secret identities for the code executed by a training run."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import re
import stat
import subprocess
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError
from importlib.metadata import distribution as _metadata_distribution
from importlib.metadata import version as _metadata_version
from importlib.util import find_spec as _stdlib_find_spec
from pathlib import Path, PurePosixPath
from typing import Any, Final


_MAX_GIT_OUTPUT_BYTES: Final = 4 * 1024 * 1024
_MAX_CHANGE_RECORDS: Final = 2_048
_MAX_CHANGED_FILE_BYTES: Final = 8 * 1024 * 1024
_MAX_CHANGED_TOTAL_BYTES: Final = 32 * 1024 * 1024
_MAX_DEPENDENCY_FILE_BYTES: Final = 2 * 1024 * 1024 * 1024
_MAX_DISTRIBUTION_RECORD_BYTES: Final = 4 * 1024 * 1024
_HASH_CHUNK_BYTES: Final = 1024 * 1024
_MAX_DRIVER_OUTPUT_BYTES: Final = 16 * 1024
_MAX_READELF_OUTPUT_BYTES: Final = 64 * 1024
_PINNED_ATTENTION_BACKEND: Final = "flash_attention_2"
_CUDA_RUNTIME_DISTRIBUTION: Final = "nvidia-cuda-runtime-cu12"
_CUDA_RUNTIME_RELATIVE_PATH: Final = "nvidia/cuda_runtime/lib/libcudart.so.12"
_CUDA_RUNTIME_SONAME: Final = "libcudart.so.12"

# Receipt component -> (installed distribution, selected module).
_DEPENDENCY_IMPORTS: Final[dict[str, tuple[str, str]]] = {
    "ms-swift": ("ms-swift", "swift"),
    "transformers": ("transformers", "transformers"),
    "flash-attn": ("flash-attn", "flash_attn"),
    "flash_attn_2_cuda": ("flash-attn", "flash_attn_2_cuda"),
    "torch": ("torch", "torch"),
    "accelerate": ("accelerate", "accelerate"),
    "peft": ("peft", "peft"),
    "tokenizers": ("tokenizers", "tokenizers"),
}

# Dependency component -> stable receipt name -> imported implementation owner.
_DEPENDENCY_SOURCE_IMPORTS: Final[dict[str, dict[str, str]]] = {
    "transformers": {
        "auto_configuration": "transformers.models.auto.configuration_auto",
        "auto_image_processing": "transformers.models.auto.image_processing_auto",
        "auto_processing": "transformers.models.auto.processing_auto",
        "auto_tokenization": "transformers.models.auto.tokenization_auto",
        "flash_attention_integration": "transformers.integrations.flash_attention",
        "flash_attention_utils": "transformers.modeling_flash_attention_utils",
        "modeling_utils": "transformers.modeling_utils",
        "optimization": "transformers.optimization",
        "qwen2_tokenization": "transformers.models.qwen2.tokenization_qwen2",
        "qwen2_tokenization_fast": "transformers.models.qwen2.tokenization_qwen2_fast",
        "qwen2_vl_image_processing_fast": (
            "transformers.models.qwen2_vl.image_processing_qwen2_vl_fast"
        ),
        "qwen3_vl_configuration": "transformers.models.qwen3_vl.configuration_qwen3_vl",
        "qwen3_vl_modeling": "transformers.models.qwen3_vl.modeling_qwen3_vl",
        "qwen3_vl_processing": "transformers.models.qwen3_vl.processing_qwen3_vl",
        "tokenization_utils": "transformers.tokenization_utils",
        "tokenization_utils_fast": "transformers.tokenization_utils_fast",
        "trainer_utils": "transformers.trainer_utils",
    },
    "accelerate": {
        "accelerator": "accelerate.accelerator",
        "modeling": "accelerate.utils.modeling",
        "operations": "accelerate.utils.operations",
        "state": "accelerate.state",
    },
    "flash-attn": {
        "flash_attn_interface": "flash_attn.flash_attn_interface",
    },
    "peft": {
        "dora_implementation": "peft.tuners.lora.dora",
        "dora_layer": "peft.tuners.lora.layer",
        "lora_config": "peft.tuners.lora.config",
        "lora_model": "peft.tuners.lora.model",
        "mapping_func": "peft.mapping_func",
        "peft_model": "peft.peft_model",
        "tuners_utils": "peft.tuners.tuners_utils",
    },
    "tokenizers": {
        "tokenizers_extension": "tokenizers.tokenizers",
    },
    "torch": {
        "torch_c_extension": "torch._C",
    },
}

_DEPENDENCY_SOURCE_SYMBOLS: Final[dict[str, dict[str, tuple[str, ...]]]] = {
    "transformers": {
        "auto_configuration": ("AutoConfig",),
        "auto_image_processing": ("AutoImageProcessor",),
        "auto_processing": ("AutoProcessor",),
        "auto_tokenization": ("AutoTokenizer", "tokenizer_class_from_name"),
        "flash_attention_integration": ("flash_attention_forward",),
        "flash_attention_utils": (
            "FlashAttentionKwargs",
            "_flash_attention_forward",
            "lazy_import_flash_attention",
            "prepare_fa_kwargs_from_position_ids",
        ),
        "modeling_utils": ("ALL_ATTENTION_FUNCTIONS",),
        "optimization": ("get_cosine_schedule_with_warmup",),
        "qwen2_tokenization": ("Qwen2Tokenizer",),
        "qwen2_tokenization_fast": ("Qwen2TokenizerFast",),
        "qwen2_vl_image_processing_fast": ("Qwen2VLImageProcessorFast",),
        "qwen3_vl_configuration": ("Qwen3VLConfig", "Qwen3VLTextConfig"),
        "qwen3_vl_modeling": (
            "Qwen3VLForConditionalGeneration",
            "Qwen3VLTextAttention",
            "Qwen3VLTextModel",
            "Qwen3VLVisionAttention",
        ),
        "qwen3_vl_processing": ("Qwen3VLProcessor",),
        "tokenization_utils": ("PreTrainedTokenizer",),
        "tokenization_utils_fast": ("PreTrainedTokenizerFast",),
        "trainer_utils": ("set_seed",),
    },
    "peft": {
        "dora_implementation": ("DoraLinearLayer",),
        "dora_layer": ("LoraLayer",),
        "lora_config": ("LoraConfig",),
        "lora_model": ("LoraModel",),
        "mapping_func": ("get_peft_model",),
        "peft_model": ("PeftModel", "get_model_status"),
        "tuners_utils": ("BaseTuner", "BaseTunerLayer"),
    },
}

# Runtime component -> stable owner name -> (distribution, distribution-relative file).
# These are the implementation-bearing CUDA/cuDNN objects used by the pinned Torch
# stack, not every shared object shipped by either distribution.
_DEPENDENCY_NATIVE_PATHS: Final[dict[str, dict[str, tuple[str, str]]]] = {
    "torch": {
        "libtorch_cuda": ("torch", "torch/lib/libtorch_cuda.so"),
        "libc10_cuda": ("torch", "torch/lib/libc10_cuda.so"),
        "cudnn": ("nvidia-cudnn-cu12", "nvidia/cudnn/lib/libcudnn.so.9"),
        "cudnn_adv": (
            "nvidia-cudnn-cu12",
            "nvidia/cudnn/lib/libcudnn_adv.so.9",
        ),
        "cudnn_cnn": (
            "nvidia-cudnn-cu12",
            "nvidia/cudnn/lib/libcudnn_cnn.so.9",
        ),
        "cudnn_engines_precompiled": (
            "nvidia-cudnn-cu12",
            "nvidia/cudnn/lib/libcudnn_engines_precompiled.so.9",
        ),
        "cudnn_engines_runtime_compiled": (
            "nvidia-cudnn-cu12",
            "nvidia/cudnn/lib/libcudnn_engines_runtime_compiled.so.9",
        ),
        "cudnn_graph": (
            "nvidia-cudnn-cu12",
            "nvidia/cudnn/lib/libcudnn_graph.so.9",
        ),
        "cudnn_heuristic": (
            "nvidia-cudnn-cu12",
            "nvidia/cudnn/lib/libcudnn_heuristic.so.9",
        ),
        "cudnn_ops": (
            "nvidia-cudnn-cu12",
            "nvidia/cudnn/lib/libcudnn_ops.so.9",
        ),
    }
}


def _pinned_source_identity(
    module: str,
    symbols: tuple[str, ...],
    sha256: str,
    size_bytes: int,
) -> dict[str, Any]:
    return {
        "module": module,
        "symbols": list(symbols),
        "symbols_available": True,
        "imported_origin": {"status": "available"},
        "origin_kind": "source",
        "sha256": {"status": "available", "value": sha256},
        "size_bytes": {"status": "available", "value": size_bytes},
    }


_PINNED_RUNTIME_DEPENDENCIES: Final[dict[str, dict[str, Any]]] = {
    "transformers": {
        "distribution": "transformers",
        "import_name": "transformers",
        "distribution_version": {"status": "available", "value": "4.57.1"},
        "distribution_record": {
            "status": "available",
            "value": {
                "sha256": "025b8186b05df7f173dff14013042262c7dacdb556e98ee5c56a182c02e42728",
                "size_bytes": 411910,
            },
        },
        "role": "runtime_dependency",
        "origin_resolution": "imported_module",
        "imported_origin": {"status": "available"},
        "origin_kind": "source",
        "sha256": {
            "status": "available",
            "value": "4ef5187b5f66c564aa575ddab9ce94342630d40fe874d0464d790c6f6b748647",
        },
        "size_bytes": {"status": "available", "value": 47000},
        "source_identities": {
            "auto_configuration": _pinned_source_identity(
                "transformers.models.auto.configuration_auto",
                ("AutoConfig",),
                "120b62cd4e13dd155762acd14cf64fc8bca17a068f5dd004ddfd30fbcb2ef4ab",
                55606,
            ),
            "auto_image_processing": _pinned_source_identity(
                "transformers.models.auto.image_processing_auto",
                ("AutoImageProcessor",),
                "34e04aa2fc7e5bab3f6e918d8e144760af5eb7c9e0b910641e09bbecf228c27b",
                39079,
            ),
            "auto_processing": _pinned_source_identity(
                "transformers.models.auto.processing_auto",
                ("AutoProcessor",),
                "e97580de189cd7e5ea3378a96e01279939f305c68a0245a9160b80cb1a938961",
                20853,
            ),
            "auto_tokenization": _pinned_source_identity(
                "transformers.models.auto.tokenization_auto",
                ("AutoTokenizer", "tokenizer_class_from_name"),
                "bd040dfb212fb20f1bef1f8fedfc3b4c100fe3b57e603c738552aebf2521c51d",
                57863,
            ),
            "flash_attention_integration": _pinned_source_identity(
                "transformers.integrations.flash_attention",
                ("flash_attention_forward",),
                "850aa63f9473391188c357454cfb6a89339394d26f6024ccbe246f00f6344144",
                3125,
            ),
            "flash_attention_utils": _pinned_source_identity(
                "transformers.modeling_flash_attention_utils",
                _DEPENDENCY_SOURCE_SYMBOLS["transformers"]["flash_attention_utils"],
                "293fe81c6bd38aac8a3bc6ae086ff21b9695b349e97c9650b1fe61e42a36d710",
                30112,
            ),
            "modeling_utils": _pinned_source_identity(
                "transformers.modeling_utils",
                ("ALL_ATTENTION_FUNCTIONS",),
                "bf1c6b2a43cf7c36fb79f37c981424dd6ae78eb863fcaa5d2a37e76c9828611d",
                308964,
            ),
            "optimization": _pinned_source_identity(
                "transformers.optimization",
                ("get_cosine_schedule_with_warmup",),
                "41b08ffb29c2691626b225ed389a694cc8e61480818214a35f215b5c39167bfb",
                39971,
            ),
            "qwen2_tokenization": _pinned_source_identity(
                "transformers.models.qwen2.tokenization_qwen2",
                ("Qwen2Tokenizer",),
                "23f05697fcaf26fe5e328abaf524f010510a2b9f4ba2e2edc9dca0a4015a09a2",
                13935,
            ),
            "qwen2_tokenization_fast": _pinned_source_identity(
                "transformers.models.qwen2.tokenization_qwen2_fast",
                ("Qwen2TokenizerFast",),
                "1025a3b86526283bd86a447f0fe2d991d09775b32d3fdcc233c9e447b0611049",
                5210,
            ),
            "qwen2_vl_image_processing_fast": _pinned_source_identity(
                "transformers.models.qwen2_vl.image_processing_qwen2_vl_fast",
                ("Qwen2VLImageProcessorFast",),
                "09bfa9b17df7c3f0c6159bc34008ee50f21d2472cd5bae7e5c21ba1ca13a423c",
                12723,
            ),
            "qwen3_vl_configuration": _pinned_source_identity(
                "transformers.models.qwen3_vl.configuration_qwen3_vl",
                ("Qwen3VLConfig", "Qwen3VLTextConfig"),
                "177fd0a4dc1b08307c08ca72cf26b8d7dc028ab6ab5975bf650fc14ab8132a83",
                14827,
            ),
            "qwen3_vl_modeling": _pinned_source_identity(
                "transformers.models.qwen3_vl.modeling_qwen3_vl",
                _DEPENDENCY_SOURCE_SYMBOLS["transformers"]["qwen3_vl_modeling"],
                "dd63ed3b124232735b3dca1bfa28f9d6b0d3f7182afcb75dde8f3e724b2b22da",
                70877,
            ),
            "qwen3_vl_processing": _pinned_source_identity(
                "transformers.models.qwen3_vl.processing_qwen3_vl",
                ("Qwen3VLProcessor",),
                "efd8d64aaf608aad1ffb3e6d503d6a99e5227d007df95c1d9fa905d998cda4a9",
                17149,
            ),
            "tokenization_utils": _pinned_source_identity(
                "transformers.tokenization_utils",
                ("PreTrainedTokenizer",),
                "dfcc42414037d865d22568259eec12dd10caa7375b2d7a27707b7a502a08ef80",
                47780,
            ),
            "tokenization_utils_fast": _pinned_source_identity(
                "transformers.tokenization_utils_fast",
                ("PreTrainedTokenizerFast",),
                "558e454ee6850e90e8bc5d1782e35878dfda1da9bfdf997ae113fa7ff771a0f3",
                41383,
            ),
            "trainer_utils": _pinned_source_identity(
                "transformers.trainer_utils",
                ("set_seed",),
                "20e32d05ef22f366ab9ce91c6dbec2290b2e46f8da749d52eeebaf6a4349bd8e",
                34254,
            ),
        },
        "native_identities": {},
    },
    "flash-attn": {
        "distribution": "flash-attn",
        "import_name": "flash_attn",
        "distribution_version": {"status": "available", "value": "2.8.3"},
        "distribution_record": {
            "status": "available",
            "value": {
                "sha256": "fee009739702fa997fa85c07f134ad54636d042475235515b90afdcfffd299a5",
                "size_bytes": 15390,
            },
        },
        "role": "runtime_dependency",
        "origin_resolution": "imported_module",
        "imported_origin": {"status": "available"},
        "origin_kind": "source",
        "sha256": {
            "status": "available",
            "value": "f1833af940ac1124e09dc05dd922308871411035e75b90bf4dbe3a831dc03b50",
        },
        "size_bytes": {"status": "available", "value": 285},
        "source_identities": {
            "flash_attn_interface": {
                "module": "flash_attn.flash_attn_interface",
                "imported_origin": {"status": "available"},
                "origin_kind": "source",
                "sha256": {
                    "status": "available",
                    "value": "e8da8127f7ebf5c5aeb7f35b316ff96394a70553378f125717ea174af912db13",
                },
                "size_bytes": {"status": "available", "value": 60677},
            }
        },
        "native_identities": {},
    },
    "flash_attn_2_cuda": {
        "distribution": "flash-attn",
        "import_name": "flash_attn_2_cuda",
        "distribution_version": {"status": "available", "value": "2.8.3"},
        "distribution_record": {
            "status": "available",
            "value": {
                "sha256": "fee009739702fa997fa85c07f134ad54636d042475235515b90afdcfffd299a5",
                "size_bytes": 15390,
            },
        },
        "role": "runtime_dependency",
        "origin_resolution": "imported_module",
        "imported_origin": {"status": "available"},
        "origin_kind": "binary",
        "sha256": {
            "status": "available",
            "value": "8ca052bf2d3f53baa629e22749b9622a95273c5bffb5f06cd24768ef63f65807",
        },
        "size_bytes": {"status": "available", "value": 997961816},
        "source_identities": {},
        "native_identities": {},
    },
    "torch": {
        "distribution": "torch",
        "import_name": "torch",
        "distribution_version": {"status": "available", "value": "2.9.1"},
        "distribution_record": {
            "status": "available",
            "value": {
                "sha256": "a8a2b13b3ba6e31a168babb44dcbdda35d3f56583ae1ba9e98ad51ae809cb0e7",
                "size_bytes": 1364780,
            },
        },
        "role": "runtime_dependency",
        "origin_resolution": "imported_module",
        "imported_origin": {"status": "available"},
        "origin_kind": "source",
        "sha256": {
            "status": "available",
            "value": "3caf7f40140ede2465bde40b9003af10cbbc8f7bcf436fa1de026471daa1b288",
        },
        "size_bytes": {"status": "available", "value": 102969},
        "source_identities": {
            "torch_c_extension": {
                "module": "torch._C",
                "imported_origin": {"status": "available"},
                "origin_kind": "binary",
                "sha256": {
                    "status": "available",
                    "value": "90fc84350d2de15ca167734eb63b20b3a1a4ea2721d498b66e3e65174c0cff9f",
                },
                "size_bytes": {"status": "available", "value": 25521},
            }
        },
        "native_identities": {
            "libtorch_cuda": {
                "distribution": "torch",
                "relative_path": "torch/lib/libtorch_cuda.so",
                "imported_origin": {"status": "available"},
                "origin_kind": "binary",
                "sha256": {
                    "status": "available",
                    "value": "02250527966ae122ccfc89d0306736874f9c619ba04431871ef76175e7253b66",
                },
                "size_bytes": {"status": "available", "value": 1022776209},
            },
            "libc10_cuda": {
                "distribution": "torch",
                "relative_path": "torch/lib/libc10_cuda.so",
                "imported_origin": {"status": "available"},
                "origin_kind": "binary",
                "sha256": {
                    "status": "available",
                    "value": "2f6027b42aee93b5db3f814b92dfbe179074cc928eebad85804b6a266214d5f7",
                },
                "size_bytes": {"status": "available", "value": 697169},
            },
            "cudnn": {
                "distribution": "nvidia-cudnn-cu12",
                "relative_path": "nvidia/cudnn/lib/libcudnn.so.9",
                "imported_origin": {"status": "available"},
                "origin_kind": "binary",
                "sha256": {
                    "status": "available",
                    "value": "3b68ea689be647bce63cb3c2d9edb589add415405a7e9f296d69ca029cae0b8b",
                },
                "size_bytes": {"status": "available", "value": 125136},
            },
            "cudnn_adv": {
                "distribution": "nvidia-cudnn-cu12",
                "relative_path": "nvidia/cudnn/lib/libcudnn_adv.so.9",
                "imported_origin": {"status": "available"},
                "origin_kind": "binary",
                "sha256": {
                    "status": "available",
                    "value": "9814d64e04fce1f6cbd16d02ccefd6611541b3c61736fb450e334b482635ca07",
                },
                "size_bytes": {"status": "available", "value": 285226600},
            },
            "cudnn_cnn": {
                "distribution": "nvidia-cudnn-cu12",
                "relative_path": "nvidia/cudnn/lib/libcudnn_cnn.so.9",
                "imported_origin": {"status": "available"},
                "origin_kind": "binary",
                "sha256": {
                    "status": "available",
                    "value": "655c2d2649466c73b6b4c19705fb6906fd95e331a1f8218314b49f37dd7cafed",
                },
                "size_bytes": {"status": "available", "value": 6301096},
            },
            "cudnn_engines_precompiled": {
                "distribution": "nvidia-cudnn-cu12",
                "relative_path": "nvidia/cudnn/lib/libcudnn_engines_precompiled.so.9",
                "imported_origin": {"status": "available"},
                "origin_kind": "binary",
                "sha256": {
                    "status": "available",
                    "value": "f504745d346542609f975de51227080647d3675eac576ead82c396b8124f8f28",
                },
                "size_bytes": {"status": "available", "value": 547383096},
            },
            "cudnn_engines_runtime_compiled": {
                "distribution": "nvidia-cudnn-cu12",
                "relative_path": "nvidia/cudnn/lib/libcudnn_engines_runtime_compiled.so.9",
                "imported_origin": {"status": "available"},
                "origin_kind": "binary",
                "sha256": {
                    "status": "available",
                    "value": "8222672743c4f2e39feb1c365d6f6b6c40f56023471506f49944910847f1ee0a",
                },
                "size_bytes": {"status": "available", "value": 23094328},
            },
            "cudnn_graph": {
                "distribution": "nvidia-cudnn-cu12",
                "relative_path": "nvidia/cudnn/lib/libcudnn_graph.so.9",
                "imported_origin": {"status": "available"},
                "origin_kind": "binary",
                "sha256": {
                    "status": "available",
                    "value": "787c955dce49091ead850e4536666594095ea9f92a8a08879d8ddad466674657",
                },
                "size_bytes": {"status": "available", "value": 4439912},
            },
            "cudnn_heuristic": {
                "distribution": "nvidia-cudnn-cu12",
                "relative_path": "nvidia/cudnn/lib/libcudnn_heuristic.so.9",
                "imported_origin": {"status": "available"},
                "origin_kind": "binary",
                "sha256": {
                    "status": "available",
                    "value": "2804eee5f6fc11d0299ba07687eda0a974d3b2436357c639706c4e282cc2cc05",
                },
                "size_bytes": {"status": "available", "value": 58726712},
            },
            "cudnn_ops": {
                "distribution": "nvidia-cudnn-cu12",
                "relative_path": "nvidia/cudnn/lib/libcudnn_ops.so.9",
                "imported_origin": {"status": "available"},
                "origin_kind": "binary",
                "sha256": {
                    "status": "available",
                    "value": "34f56d67f3df108949d1e5e03543064e055e217d0f5e53dc6d4f7c9454d852f8",
                },
                "size_bytes": {"status": "available", "value": 127936080},
            },
        },
    },
    "cuda-runtime": {
        "distribution": "nvidia-cuda-runtime-cu12",
        "import_name": None,
        "distribution_version": {"status": "available", "value": "12.8.90"},
        "distribution_record": {
            "status": "available",
            "value": {
                "sha256": "bef69015da09064656ac31d35fc73d6436712380c602ff38fb5fc16c30512bdf",
                "size_bytes": 11369,
            },
        },
        "role": "runtime_dependency",
        "origin_resolution": "loaded_shared_object",
        "loaded_soname": "libcudart.so.12",
        "distribution_relative_path": "nvidia/cuda_runtime/lib/libcudart.so.12",
        "distribution_origin": {"status": "available"},
        "imported_origin": {"status": "available"},
        "loaded_origin_matches_distribution": True,
        "origin_kind": "binary",
        "sha256": {
            "status": "available",
            "value": "c3a75b33af334a3486d197dbd1584a2985183ba4688d237a2be5f2f679329920",
        },
        "size_bytes": {"status": "available", "value": 728800},
        "source_repository": {
            "status": "unavailable",
            "reason": "not_source_origin",
        },
        "source_identities": {},
        "native_identities": {},
    },
    "accelerate": {
        "distribution": "accelerate",
        "import_name": "accelerate",
        "distribution_version": {"status": "available", "value": "1.10.1"},
        "distribution_record": {
            "status": "available",
            "value": {
                "sha256": "8c84d01d529cd3d9745936fd74933835e9838e7119a96832a27123e5170500f8",
                "size_bytes": 13939,
            },
        },
        "role": "runtime_dependency",
        "origin_resolution": "imported_module",
        "imported_origin": {"status": "available"},
        "origin_kind": "source",
        "sha256": {
            "status": "available",
            "value": "68f56baac9b078db4735567649441ef28932a946b400bf8d8c2519b0a5b94b89",
        },
        "size_bytes": {"status": "available", "value": 1555},
        "source_identities": {
            "accelerator": {
                "module": "accelerate.accelerator",
                "imported_origin": {"status": "available"},
                "origin_kind": "source",
                "sha256": {
                    "status": "available",
                    "value": "9005e9a74cd819701578ea9a86e8cd71c16152f0e2308ed1b6f447c0797837f9",
                },
                "size_bytes": {"status": "available", "value": 190652},
            },
            "operations": {
                "module": "accelerate.utils.operations",
                "imported_origin": {"status": "available"},
                "origin_kind": "source",
                "sha256": {
                    "status": "available",
                    "value": "f384a95f02ac6cd1c778f6e82fd814a8ae9b039be8d24c87fe02c07709f313fb",
                },
                "size_bytes": {"status": "available", "value": 31266},
            },
            "modeling": {
                "module": "accelerate.utils.modeling",
                "imported_origin": {"status": "available"},
                "origin_kind": "source",
                "sha256": {
                    "status": "available",
                    "value": "35cbf0f316b086f599bf54babf4d7691ca66fd675ef1dfcfb10821bd5e983a02",
                },
                "size_bytes": {"status": "available", "value": 95815},
            },
            "state": {
                "module": "accelerate.state",
                "imported_origin": {"status": "available"},
                "origin_kind": "source",
                "sha256": {
                    "status": "available",
                    "value": "dfa76df4d4205babb1f8c1ce87206d359168f28d551b8851b64f159a4911537a",
                },
                "size_bytes": {"status": "available", "value": 57976},
            },
        },
        "native_identities": {},
    },
    "peft": {
        "distribution": "peft",
        "import_name": "peft",
        "distribution_version": {"status": "available", "value": "0.17.1"},
        "distribution_record": {
            "status": "available",
            "value": {
                "sha256": "258749eb655ec7ee9b6d4f6040b2a34c9207cafc9f3efc35d96559e63460a38c",
                "size_bytes": 22735,
            },
        },
        "role": "runtime_dependency",
        "origin_resolution": "imported_module",
        "imported_origin": {"status": "available"},
        "origin_kind": "source",
        "sha256": {
            "status": "available",
            "value": "efabc3b44d7326eee59c07910e426fa38ee0c6419b773e8f75af87ae61d3fa32",
        },
        "size_bytes": {"status": "available", "value": 5523},
        "source_identities": {
            "dora_implementation": _pinned_source_identity(
                "peft.tuners.lora.dora",
                ("DoraLinearLayer",),
                "07d514fd057d8aa9ec7b89bec3f3fbd04862951c1a8ff73153e833096481ecf5",
                8480,
            ),
            "dora_layer": _pinned_source_identity(
                "peft.tuners.lora.layer",
                ("LoraLayer",),
                "39e2f6908c9f3faf4d2de2b6ecfdf44c96d2cae67b8155ed1b51bc67fb25b734",
                97115,
            ),
            "lora_config": _pinned_source_identity(
                "peft.tuners.lora.config",
                ("LoraConfig",),
                "85a478bcdfd42f9398d232d5e0ddb49650bd1199148503f864834fdbeca3aebb",
                42385,
            ),
            "lora_model": _pinned_source_identity(
                "peft.tuners.lora.model",
                ("LoraModel",),
                "9eec396e506134e50afa41ff21b2608a3307f8f07207f460506d2da33d7bc2f7",
                45130,
            ),
            "mapping_func": _pinned_source_identity(
                "peft.mapping_func",
                ("get_peft_model",),
                "c30533cb009ea60e45a35d564285be65c9778c0c41e0c281dc2dfacd69c2315d",
                6064,
            ),
            "peft_model": _pinned_source_identity(
                "peft.peft_model",
                ("PeftModel", "get_model_status"),
                "2307ebeb101baff53b19ce5a0f109885d99b64ab147d67cba19719b6097c3e10",
                156704,
            ),
            "tuners_utils": _pinned_source_identity(
                "peft.tuners.tuners_utils",
                ("BaseTuner", "BaseTunerLayer"),
                "d5b6a92d8f0b5325951519e81d2ed9debcba6ace1028e887958dd5e5188be66a",
                67995,
            ),
        },
        "native_identities": {},
    },
    "tokenizers": {
        "distribution": "tokenizers",
        "import_name": "tokenizers",
        "distribution_version": {"status": "available", "value": "0.22.0"},
        "distribution_record": {
            "status": "available",
            "value": {
                "sha256": "2df0edacccd5e83cc9fde45cc5199701bc97731a676787ada63d02b8ff0f4bf0",
                "size_bytes": 3650,
            },
        },
        "role": "runtime_dependency",
        "origin_resolution": "imported_module",
        "imported_origin": {"status": "available"},
        "origin_kind": "source",
        "sha256": {
            "status": "available",
            "value": "644e596a052fa1b05272b1c141d1286e1c78c2a3346ecabd17d68b62404d8d84",
        },
        "size_bytes": {"status": "available", "value": 2615},
        "source_identities": {
            "tokenizers_extension": {
                "module": "tokenizers.tokenizers",
                "imported_origin": {"status": "available"},
                "origin_kind": "binary",
                "sha256": {
                    "status": "available",
                    "value": "1e1a657d75b22975395c66f0ce1b341fe9eec4b4c17a9e0da3c93a2ffddbd6db",
                },
                "size_bytes": {"status": "available", "value": 10028880},
            }
        },
        "native_identities": {},
    },
}

_PINNED_NATIVE_BUILD_IDS: Final[dict[str, str]] = {
    "libtorch_cuda": "bf4c31d86c74fe1e5d5c67caea2d1a006785ce95",
    "libc10_cuda": "d9bfdd5a168655341fb2595f9dd763c9be969281",
    "cudnn": "989afd40d5f56a848608b792eda320d3c4067d4c",
    "cudnn_adv": "40dad754fc3521e5a5531381d3aecc1e0ba31455",
    "cudnn_cnn": "4d8c00415c2e0f9b967a6614588acad6bc5b3164",
    "cudnn_engines_precompiled": "0e111592d57f71d0ccc36621f67c89d9197ec74d",
    "cudnn_engines_runtime_compiled": "f8099f58a92ca1e2671c13927faef534fa9beab7",
    "cudnn_graph": "429aed8e6a0dcc9974a1371b071b8b4d590d0545",
    "cudnn_heuristic": "11532cd128025c2c19cfceea30702b0176035ea9",
    "cudnn_ops": "8ca23749d0fd44f406607efb7f09986da436204c",
}
for _native_name, _native_build_id in _PINNED_NATIVE_BUILD_IDS.items():
    _PINNED_RUNTIME_DEPENDENCIES["torch"]["native_identities"][_native_name][
        "elf_build_id"
    ] = {"status": "available", "value": _native_build_id}
_PINNED_RUNTIME_DEPENDENCIES["cuda-runtime"]["elf_build_id"] = {
    "status": "available",
    "value": "7b1714ea2d766ca35afe1e3dd34a75b41b78999f",
}

_PINNED_DRIVER_NATIVE_IDENTITY: Final[dict[str, Any]] = {
    "soname_family": "libcuda.so",
    "origin": "/usr/lib/x86_64-linux-gnu/libcuda.so.550.54.15",
    "sha256": "685a6623d4d574442e94d748f040b1fd36edf665f1e10f17ed9efc2efa74acf6",
    "size_bytes": 28392536,
    "elf_build_id": "6451f06a1c8a877b03a1523720126f0421b57293",
}

_PINNED_REFERENCE_ONLY: Final[dict[str, dict[str, Any]]] = {
    "ms-swift": {
        "distribution": "ms-swift",
        "import_name": "swift",
        "distribution_version": {"status": "available", "value": "4.2.2"},
        "role": "reference_only_not_imported_by_training_route",
        "origin_resolution": "import_spec_without_import",
        "imported_origin": {
            "status": "unavailable",
            "reason": "reference_only_not_imported",
        },
        "origin_kind": "source",
        "sha256": {
            "status": "available",
            "value": "d345fd8f68077d11730ffe56747a52b1858550e8c21067ed55a1db2f79ab5caf",
        },
        "size_bytes": {"status": "available", "value": 3529},
        "source_repository": {
            "status": "available",
            "value": {
                "commit": "f2797138dba0e224cfff735cd89a528a08d8732a",
                "state": "clean",
            },
        },
    }
}

_PINNED_RUNTIME: Final[dict[str, Any]] = {
    "python": {"implementation": "CPython", "version": "3.12.11"},
    "torch_cuda": {
        "torch_version": "2.9.1+cu128",
        "torch_git_version": "5811a8d7da873dd699ff6687092c225caffcf1bb",
        "cuda_compiled_version": "12.8",
        "hip_compiled_version": None,
        "cuda_available": True,
        "cudnn_version": 91002,
        "nvidia_driver_version": {"status": "available", "value": "550.54.15"},
        "cuda_driver_runtime_applicable": True,
    },
}

_PINNED_RUNTIME_BASELINE: Final[dict[str, Any]] = {
    "schema_version": 3,
    "attention_backend": _PINNED_ATTENTION_BACKEND,
    "dependencies": _PINNED_RUNTIME_DEPENDENCIES,
    "runtime": _PINNED_RUNTIME,
    "reference_only": _PINNED_REFERENCE_ONLY,
}
PINNED_RUNTIME_BASELINE_SHA256: Final = hashlib.sha256(
    json.dumps(
        _PINNED_RUNTIME_BASELINE,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()

_DEPENDENCY_FILES = {
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "environment.yml",
    "environment.yaml",
    "conda-lock.yml",
    "conda-lock.yaml",
    "poetry.lock",
    "uv.lock",
    "pdm.lock",
}


class PinnedRuntimeAdmissionError(RuntimeError):
    """Raised when observed runtime identity does not match the accepted stack."""

    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        mismatch_paths = ", ".join(result["mismatches"])
        super().__init__(f"pinned runtime baseline admission failed: {mismatch_paths}")


class NativeExecutionAttestationError(RuntimeError):
    """Raised when CUDA-initialized mapped native objects are not admitted."""

    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        mismatch_paths = ", ".join(result["mismatches"])
        super().__init__(
            f"mapped native execution attestation failed: {mismatch_paths}"
        )


def pinned_runtime_baseline() -> dict[str, Any]:
    """Return an isolated JSON-compatible copy of the accepted runtime baseline."""

    return json.loads(
        json.dumps(
            _PINNED_RUNTIME_BASELINE,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def compare_pinned_runtime_baseline(
    *,
    provenance: Mapping[str, Any],
    attention_backend: str,
) -> dict[str, Any]:
    """Compare pre-model provenance with the exact accepted runtime identity.

    Absolute import paths and CUDA device count remain recorded observations, not
    portable admission keys. Runtime dependency content, package manifests,
    named implementation owners, Python/Torch/CUDA/cuDNN/driver facts, and the
    configured attention backend are exact admission keys. The selected
    ms-swift checkout is reported independently and never affects admission.
    """

    mismatches: list[str] = []
    if attention_backend != _PINNED_ATTENTION_BACKEND:
        mismatches.append("attention_backend")

    dependencies = provenance.get("dependencies")
    runtime = provenance.get("runtime")
    _compare_expected(
        _PINNED_RUNTIME_DEPENDENCIES,
        dependencies,
        path="dependencies",
        mismatches=mismatches,
    )
    _compare_expected(
        _PINNED_RUNTIME,
        runtime,
        path="runtime",
        mismatches=mismatches,
    )

    observed_runtime_components: set[str] = set()
    if isinstance(dependencies, Mapping):
        for name, component in dependencies.items():
            if name in _PINNED_RUNTIME_DEPENDENCIES:
                continue
            if (
                isinstance(component, Mapping)
                and component.get("role") == "runtime_dependency"
            ):
                observed_runtime_components.add(str(name))
    for name in sorted(observed_runtime_components):
        mismatches.append(f"dependencies.{name}")

    if isinstance(dependencies, Mapping):
        for component_name, expected_component in _PINNED_RUNTIME_DEPENDENCIES.items():
            observed_component = dependencies.get(component_name)
            if not isinstance(observed_component, Mapping):
                continue
            for owner_group in ("source_identities", "native_identities"):
                expected_owners = expected_component.get(owner_group)
                observed_owners = observed_component.get(owner_group)
                if not isinstance(expected_owners, Mapping) or not isinstance(
                    observed_owners, Mapping
                ):
                    continue
                unknown_owners = set(observed_owners) - set(expected_owners)
                for owner_name in sorted(
                    unknown_owners,
                    key=lambda value: (type(value).__name__, str(value)),
                ):
                    mismatches.append(
                        f"dependencies.{component_name}.{owner_group}.{owner_name}"
                    )

    reference_result: dict[str, Any] = {}
    for name, expected in _PINNED_REFERENCE_ONLY.items():
        reference_mismatches: list[str] = []
        observed = dependencies.get(name) if isinstance(dependencies, Mapping) else None
        _compare_expected(
            expected,
            observed,
            path=f"dependencies.{name}",
            mismatches=reference_mismatches,
        )
        reference_result[name] = {
            "matches_recorded_reference": not reference_mismatches,
            "mismatches": sorted(set(reference_mismatches)),
        }

    unique_mismatches = sorted(set(mismatches))
    return {
        "schema_version": 3,
        "baseline_sha256": PINNED_RUNTIME_BASELINE_SHA256,
        "attention_backend": attention_backend,
        "admitted": not unique_mismatches,
        "mismatches": unique_mismatches,
        "reference_only": reference_result,
    }


def require_pinned_runtime_baseline(
    *,
    provenance: Mapping[str, Any],
    attention_backend: str,
) -> dict[str, Any]:
    """Return the comparison receipt or raise before cache/model work."""

    result = compare_pinned_runtime_baseline(
        provenance=provenance,
        attention_backend=attention_backend,
    )
    if not result["admitted"]:
        raise PinnedRuntimeAdmissionError(result)
    return result


def collect_mapped_native_execution_attestation(
    *,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Attest CUDA-related DSOs actually mapped after CUDA initialization."""

    if not _cuda_runtime_is_initialized():
        return {
            "schema_version": 1,
            "cuda_initialized": False,
            "admitted": False,
            "mismatches": ["cuda_initialized"],
            "components": {},
            "mapped_cudnn_components": [],
        }

    mappings = _mapped_shared_object_origins()
    mismatches: list[str] = []
    components: dict[str, Any] = {}
    dependencies = provenance.get("dependencies")
    if not isinstance(dependencies, Mapping):
        dependencies = {}

    fixed_preloads = {
        "libtorch_cuda": _nested_mapping(
            dependencies, "torch", "native_identities", "libtorch_cuda"
        ),
        "libc10_cuda": _nested_mapping(
            dependencies, "torch", "native_identities", "libc10_cuda"
        ),
        "libcudart": _nested_mapping(dependencies, "cuda-runtime"),
    }
    fixed_sonames = {
        "libtorch_cuda": "libtorch_cuda.so",
        "libc10_cuda": "libc10_cuda.so",
        "libcudart": "libcudart.so.12",
    }
    for component_name, soname in fixed_sonames.items():
        loaded = _one_mapped_origin(mappings, soname)
        components[component_name] = _attest_mapped_preload_component(
            component_name=component_name,
            loaded_origin=loaded,
            preload=fixed_preloads[component_name],
            mismatches=mismatches,
        )

    cudnn_preloads = _nested_mapping(dependencies, "torch", "native_identities")
    cudnn_by_soname: dict[str, tuple[str, Mapping[str, Any]]] = {}
    for owner_name, preload in cudnn_preloads.items():
        if not str(owner_name).startswith("cudnn") or not isinstance(preload, Mapping):
            continue
        relative_path = preload.get("relative_path")
        if isinstance(relative_path, str):
            cudnn_by_soname[Path(relative_path).name] = (str(owner_name), preload)

    mapped_cudnn_components: list[str] = []
    mapped_cudnn_sonames = sorted(
        name for name in mappings if name.startswith("libcudnn") and ".so.9" in name
    )
    if "libcudnn.so.9" not in mapped_cudnn_sonames:
        mismatches.append("components.libcudnn.mapped_origin")
    for soname in mapped_cudnn_sonames:
        expected = cudnn_by_soname.get(soname)
        if expected is None:
            mismatches.append(f"components.cudnn_unknown.{soname}")
            continue
        preload_owner_name, preload = expected
        component_name = f"lib{preload_owner_name}"
        loaded = _one_mapped_origin(mappings, soname)
        components[component_name] = _attest_mapped_preload_component(
            component_name=component_name,
            loaded_origin=loaded,
            preload=preload,
            mismatches=mismatches,
        )
        mapped_cudnn_components.append(component_name)

    driver_candidates = sorted(
        {
            origin
            for soname, origins in mappings.items()
            if soname.startswith("libcuda.so") and soname != "libcudart.so.12"
            for origin in origins
        },
        key=str,
    )
    driver_origin = driver_candidates[0] if len(driver_candidates) == 1 else None
    components["libcuda"] = _attest_pinned_driver(
        driver_origin,
        mismatches=mismatches,
    )

    unique_mismatches = sorted(set(mismatches))
    return {
        "schema_version": 1,
        "cuda_initialized": True,
        "admitted": not unique_mismatches,
        "mismatches": unique_mismatches,
        "components": components,
        "mapped_cudnn_components": sorted(mapped_cudnn_components),
    }


def require_mapped_native_execution_attestation(
    *,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    result = collect_mapped_native_execution_attestation(provenance=provenance)
    if not result["admitted"]:
        raise NativeExecutionAttestationError(result)
    return result


def _nested_mapping(owner: Mapping[str, Any], *path: str) -> Mapping[str, Any]:
    current: object = owner
    for part in path:
        if not isinstance(current, Mapping):
            return {}
        current = current.get(part)
    return current if isinstance(current, Mapping) else {}


def _one_mapped_origin(
    mappings: Mapping[str, tuple[Path, ...]],
    soname: str,
) -> Path | None:
    origins = mappings.get(soname, ())
    return origins[0] if len(origins) == 1 else None


def _attest_mapped_preload_component(
    *,
    component_name: str,
    loaded_origin: Path | None,
    preload: Mapping[str, Any],
    mismatches: list[str],
) -> dict[str, Any]:
    path_prefix = f"components.{component_name}"
    if loaded_origin is None:
        mismatches.append(f"{path_prefix}.mapped_origin")
        return {"mapped_origin": _unavailable("mapped_origin_unavailable")}
    observed = _native_file_identity(loaded_origin)
    preload_origin = preload.get("distribution_origin", preload.get("imported_origin"))
    preload_path = _available_path(preload_origin)
    same_file = False
    if preload_path is not None:
        try:
            same_file = loaded_origin.samefile(preload_path)
        except OSError:
            same_file = False
    observed["origin_matches_preload"] = same_file
    if not same_file:
        mismatches.append(f"{path_prefix}.origin_matches_preload")
    for field in ("sha256", "size_bytes", "elf_build_id"):
        field_mismatches: list[str] = []
        _compare_expected(
            preload.get(field),
            observed.get(field),
            path=f"{path_prefix}.{field}",
            mismatches=field_mismatches,
        )
        mismatches.extend(field_mismatches)
    return observed


def _attest_pinned_driver(
    loaded_origin: Path | None,
    *,
    mismatches: list[str],
) -> dict[str, Any]:
    path_prefix = "components.libcuda"
    if loaded_origin is None:
        mismatches.append(f"{path_prefix}.mapped_origin")
        return {"mapped_origin": _unavailable("mapped_origin_unavailable")}
    observed = _native_file_identity(loaded_origin)
    expected = {
        "mapped_origin": _available(_PINNED_DRIVER_NATIVE_IDENTITY["origin"]),
        "sha256": _available(_PINNED_DRIVER_NATIVE_IDENTITY["sha256"]),
        "size_bytes": _available(_PINNED_DRIVER_NATIVE_IDENTITY["size_bytes"]),
        "elf_build_id": _available(_PINNED_DRIVER_NATIVE_IDENTITY["elf_build_id"]),
    }
    _compare_expected(expected, observed, path=path_prefix, mismatches=mismatches)
    return observed


def _native_file_identity(path: Path) -> dict[str, Any]:
    return {
        "mapped_origin": _available(str(path)),
        "sha256": _sha256_path(path, max_bytes=_MAX_DEPENDENCY_FILE_BYTES),
        "size_bytes": _path_size(path, max_bytes=_MAX_DEPENDENCY_FILE_BYTES),
        "elf_build_id": _elf_build_id(path),
    }


def _available_path(identity: object) -> Path | None:
    if not isinstance(identity, Mapping) or identity.get("status") != "available":
        return None
    value = identity.get("value")
    if not isinstance(value, str):
        return None
    try:
        return Path(value).resolve(strict=True)
    except (OSError, RuntimeError):
        return None


def _compare_expected(
    expected: object,
    observed: object,
    *,
    path: str,
    mismatches: list[str],
) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(observed, Mapping):
            mismatches.append(path)
            return
        for key, expected_value in expected.items():
            child_path = f"{path}.{key}"
            if key not in observed:
                if isinstance(expected_value, Mapping):
                    _compare_expected(
                        expected_value,
                        {},
                        path=child_path,
                        mismatches=mismatches,
                    )
                else:
                    mismatches.append(child_path)
                continue
            _compare_expected(
                expected_value,
                observed[key],
                path=child_path,
                mismatches=mismatches,
            )
        return
    if observed != expected or type(observed) is not type(expected):
        mismatches.append(path)


def collect_execution_provenance(*, repository_root: str | Path) -> dict[str, Any]:
    """Return a deterministic, strict-JSON-compatible execution identity."""

    dependencies = collect_dependency_provenance()
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "repository": collect_repository_provenance(repository_root),
        "dependencies": dependencies,
        "runtime": collect_runtime_metadata(dependencies),
    }
    # Fail locally if a future field introduces NaN, a Path, or another value
    # that normal artifact writers cannot encode as strict JSON.
    json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return receipt


def collect_repository_provenance(repository_root: str | Path) -> dict[str, Any]:
    """Identify HEAD and execution-relevant working-tree changes without patches."""

    requested_root = Path(repository_root)
    try:
        probe = _run_git(["rev-parse", "--show-toplevel"], cwd=requested_root)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return _unavailable_repository("git_unavailable")
    except (OSError, subprocess.SubprocessError):
        return _unavailable_repository("not_a_git_repository")
    if probe.returncode != 0:
        return _unavailable_repository("not_a_git_repository")
    if len(probe.stdout) > _MAX_GIT_OUTPUT_BYTES:
        return _unavailable_repository("git_output_too_large")

    try:
        git_root = Path(os.fsdecode(probe.stdout.strip())).resolve(strict=True)
    except (OSError, ValueError):
        return _unavailable_repository("repository_root_unavailable")

    commit = _git_commit(git_root)
    try:
        status_result = _run_git(
            [
                "status",
                "--porcelain=v1",
                "-z",
                "--untracked-files=all",
                "--no-renames",
            ],
            cwd=git_root,
        )
    except (
        FileNotFoundError,
        subprocess.TimeoutExpired,
        OSError,
        subprocess.SubprocessError,
    ):
        return {
            "commit": commit,
            "state": "unavailable",
            "tracked_changes_present": None,
            "untracked_changes_present": None,
            "execution_relevant_changes": _empty_change_metadata(),
            "execution_relevant_digest": _unavailable("git_status_unavailable"),
        }
    if (
        status_result.returncode != 0
        or len(status_result.stdout) > _MAX_GIT_OUTPUT_BYTES
    ):
        reason = (
            "git_output_too_large"
            if len(status_result.stdout) > _MAX_GIT_OUTPUT_BYTES
            else "git_status_unavailable"
        )
        return {
            "commit": commit,
            "state": "unavailable",
            "tracked_changes_present": None,
            "untracked_changes_present": None,
            "execution_relevant_changes": _empty_change_metadata(),
            "execution_relevant_digest": _unavailable(reason),
        }

    records = _parse_git_status(status_result.stdout)
    if records is None:
        return {
            "commit": commit,
            "state": "unavailable",
            "tracked_changes_present": None,
            "untracked_changes_present": None,
            "execution_relevant_changes": _empty_change_metadata(),
            "execution_relevant_digest": _unavailable("git_status_parse_failed"),
        }

    tracked_present = any(status != "??" for status, _path in records)
    untracked_present = any(status == "??" for status, _path in records)
    relevant = [
        (status, path, path_class)
        for status, path in records
        if (path_class := _execution_path_class(path)) is not None
    ]
    relevant.sort(
        key=lambda item: (item[1].encode("utf-8", "surrogateescape"), item[0])
    )
    truncated = len(relevant) > _MAX_CHANGE_RECORDS
    bounded_relevant = relevant[:_MAX_CHANGE_RECORDS]
    path_classes: dict[str, dict[str, int]] = {}
    for status, _path, path_class in bounded_relevant:
        counts = path_classes.setdefault(path_class, {"tracked": 0, "untracked": 0})
        counts["untracked" if status == "??" else "tracked"] += 1
    path_classes = {key: path_classes[key] for key in sorted(path_classes)}
    change_metadata = {
        "count": len(bounded_relevant),
        "path_classes": path_classes,
        "truncated": truncated,
    }
    digest = (
        _unavailable("too_many_execution_relevant_changes")
        if truncated
        else _digest_relevant_changes(git_root, bounded_relevant)
    )
    return {
        "commit": commit,
        "state": "dirty" if records else "clean",
        "tracked_changes_present": tracked_present,
        "untracked_changes_present": untracked_present,
        "execution_relevant_changes": change_metadata,
        "execution_relevant_digest": digest,
    }


def collect_dependency_provenance() -> dict[str, dict[str, Any]]:
    """Collect runtime imports and reference-only dependencies independently."""

    receipt: dict[str, dict[str, Any]] = {}
    for component, (distribution, import_name) in _DEPENDENCY_IMPORTS.items():
        distribution_version = _resolve_distribution_version(distribution)
        distribution_record = _resolve_distribution_record(distribution)
        common = {
            "distribution": distribution,
            "import_name": import_name,
            "distribution_version": distribution_version,
            "distribution_record": distribution_record,
        }
        if component == "ms-swift":
            receipt[component] = {
                **common,
                **_collect_reference_dependency(import_name),
            }
        else:
            receipt[component] = {
                **common,
                **_collect_runtime_dependency(import_name),
                "source_identities": _collect_dependency_source_identities(component),
                "native_identities": _collect_dependency_native_identities(component),
            }
    receipt["cuda-runtime"] = _collect_loaded_cuda_runtime_dependency()
    return receipt


def _collect_loaded_cuda_runtime_dependency() -> dict[str, Any]:
    """Bind the installed CUDA runtime distribution to the DSO in this process."""

    distribution_origin = _distribution_file_path(
        _CUDA_RUNTIME_DISTRIBUTION,
        _CUDA_RUNTIME_RELATIVE_PATH,
    )
    loaded_origin = _loaded_shared_object_origin(_CUDA_RUNTIME_SONAME)
    origin_matches_distribution = False
    if distribution_origin is not None and loaded_origin is not None:
        try:
            origin_matches_distribution = loaded_origin.samefile(distribution_origin)
        except OSError:
            origin_matches_distribution = False

    unavailable_loaded = _unavailable("loaded_shared_object_unavailable")
    return {
        "distribution": _CUDA_RUNTIME_DISTRIBUTION,
        "import_name": None,
        "distribution_version": _resolve_distribution_version(
            _CUDA_RUNTIME_DISTRIBUTION
        ),
        "distribution_record": _resolve_distribution_record(_CUDA_RUNTIME_DISTRIBUTION),
        "role": "runtime_dependency",
        "origin_resolution": "loaded_shared_object",
        "loaded_soname": _CUDA_RUNTIME_SONAME,
        "distribution_relative_path": _CUDA_RUNTIME_RELATIVE_PATH,
        "distribution_origin": (
            _available(str(distribution_origin))
            if distribution_origin is not None
            else _unavailable("origin_unavailable")
        ),
        "imported_origin": (
            _available(str(loaded_origin))
            if loaded_origin is not None
            else unavailable_loaded
        ),
        "loaded_origin_matches_distribution": origin_matches_distribution,
        "origin_kind": "binary" if loaded_origin is not None else "unavailable",
        "sha256": (
            _sha256_path(loaded_origin, max_bytes=_MAX_DEPENDENCY_FILE_BYTES)
            if loaded_origin is not None
            else unavailable_loaded
        ),
        "size_bytes": (
            _path_size(loaded_origin, max_bytes=_MAX_DEPENDENCY_FILE_BYTES)
            if loaded_origin is not None
            else unavailable_loaded
        ),
        "elf_build_id": (
            _elf_build_id(loaded_origin)
            if loaded_origin is not None
            else unavailable_loaded
        ),
        "source_repository": _unavailable("not_source_origin"),
        "source_identities": {},
        "native_identities": {},
    }


def _collect_reference_dependency(import_name: str) -> dict[str, Any]:
    """Resolve a comparison dependency without executing its import surface."""

    origin = _reference_module_origin(import_name)
    if origin is None:
        selected_origin = _unavailable("reference_spec_resolution_failed")
        sha256 = _unavailable("reference_spec_resolution_failed")
        origin_kind = "unavailable"
        source_repository = _unavailable("reference_origin_unavailable")
    else:
        selected_origin = _available(str(origin))
        sha256 = _sha256_path(origin, max_bytes=_MAX_DEPENDENCY_FILE_BYTES)
        size_bytes = _path_size(origin, max_bytes=_MAX_DEPENDENCY_FILE_BYTES)
        origin_kind = _origin_kind(origin)
        source_repository = (
            _source_repository_identity(origin)
            if origin_kind == "source"
            else _unavailable("not_source_origin")
        )
    return {
        "role": "reference_only_not_imported_by_training_route",
        "origin_resolution": "import_spec_without_import",
        "selected_module_origin": selected_origin,
        "imported_origin": _unavailable("reference_only_not_imported"),
        "origin_kind": origin_kind,
        "sha256": sha256,
        "size_bytes": (
            size_bytes
            if origin is not None
            else _unavailable("reference_spec_resolution_failed")
        ),
        "source_repository": source_repository,
    }


def _collect_runtime_dependency(import_name: str) -> dict[str, Any]:
    origin: Path | None = None
    try:
        module = _import_module(import_name)
    except Exception:
        imported_origin = _unavailable("import_failed")
        sha256 = _unavailable("import_failed")
        size_bytes = _unavailable("import_failed")
        origin_kind = "unavailable"
    else:
        origin = _module_origin(module)
        if origin is None:
            imported_origin = _unavailable("origin_unavailable")
            sha256 = _unavailable("origin_unavailable")
            size_bytes = _unavailable("origin_unavailable")
            origin_kind = "unavailable"
        else:
            origin_kind = _origin_kind(origin)
            imported_origin = _available(str(origin))
            sha256 = _sha256_path(origin, max_bytes=_MAX_DEPENDENCY_FILE_BYTES)
            size_bytes = _path_size(origin, max_bytes=_MAX_DEPENDENCY_FILE_BYTES)
    return {
        "role": "runtime_dependency",
        "origin_resolution": "imported_module",
        "imported_origin": imported_origin,
        "origin_kind": origin_kind,
        "sha256": sha256,
        "size_bytes": size_bytes,
        "source_repository": (
            _source_repository_identity(origin)
            if origin is not None and origin_kind == "source"
            else _unavailable("not_source_origin")
        ),
    }


def _collect_dependency_source_identities(
    component: str,
) -> dict[str, dict[str, Any]]:
    identities: dict[str, dict[str, Any]] = {}
    for name, import_name in _DEPENDENCY_SOURCE_IMPORTS.get(component, {}).items():
        identities[name] = _collect_imported_source_identity(
            import_name,
            symbols=_DEPENDENCY_SOURCE_SYMBOLS.get(component, {}).get(name, ()),
        )
    return identities


def _collect_imported_source_identity(
    import_name: str,
    *,
    symbols: tuple[str, ...] = (),
) -> dict[str, Any]:
    try:
        module = _import_module(import_name)
    except Exception:
        return {
            "module": import_name,
            "symbols": list(symbols),
            "symbols_available": False,
            "imported_origin": _unavailable("import_failed"),
            "origin_kind": "unavailable",
            "sha256": _unavailable("import_failed"),
            "size_bytes": _unavailable("import_failed"),
        }
    origin = _module_origin(module)
    if origin is None:
        return {
            "module": import_name,
            "symbols": list(symbols),
            "symbols_available": False,
            "imported_origin": _unavailable("origin_unavailable"),
            "origin_kind": "unavailable",
            "sha256": _unavailable("origin_unavailable"),
            "size_bytes": _unavailable("origin_unavailable"),
        }
    return {
        "module": import_name,
        "symbols": list(symbols),
        "symbols_available": all(hasattr(module, symbol) for symbol in symbols),
        "imported_origin": _available(str(origin)),
        "origin_kind": _origin_kind(origin),
        "sha256": _sha256_path(origin, max_bytes=_MAX_DEPENDENCY_FILE_BYTES),
        "size_bytes": _path_size(origin, max_bytes=_MAX_DEPENDENCY_FILE_BYTES),
    }


def _collect_dependency_native_identities(
    component: str,
) -> dict[str, dict[str, Any]]:
    identities: dict[str, dict[str, Any]] = {}
    for name, (distribution, relative_path) in _DEPENDENCY_NATIVE_PATHS.get(
        component, {}
    ).items():
        identities[name] = _collect_distribution_file_identity(
            distribution=distribution,
            relative_path=relative_path,
        )
    return identities


def _collect_distribution_file_identity(
    *, distribution: str, relative_path: str
) -> dict[str, Any]:
    try:
        origin = _distribution_file_path(distribution, relative_path)
    except Exception:
        origin = None
    if origin is None:
        return {
            "distribution": distribution,
            "relative_path": relative_path,
            "imported_origin": _unavailable("origin_unavailable"),
            "origin_kind": "unavailable",
            "sha256": _unavailable("origin_unavailable"),
            "size_bytes": _unavailable("origin_unavailable"),
            "elf_build_id": _unavailable("origin_unavailable"),
        }
    return {
        "distribution": distribution,
        "relative_path": relative_path,
        "imported_origin": _available(str(origin)),
        "origin_kind": _origin_kind(origin),
        "sha256": _sha256_path(origin, max_bytes=_MAX_DEPENDENCY_FILE_BYTES),
        "size_bytes": _path_size(origin, max_bytes=_MAX_DEPENDENCY_FILE_BYTES),
        "elf_build_id": _elf_build_id(origin),
    }


def collect_runtime_metadata(
    dependencies: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Collect bounded Python and Torch/CUDA facts without environment dumps."""

    runtime: dict[str, Any] = {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        }
    }
    torch_receipt = dependencies.get("torch", {})
    origin = torch_receipt.get("imported_origin", {})
    if not isinstance(origin, dict) or origin.get("status") != "available":
        runtime["torch_cuda"] = _unavailable("torch_import_failed")
        return runtime
    try:
        torch_module = _import_module("torch")
    except Exception:
        runtime["torch_cuda"] = _unavailable("torch_import_failed")
        return runtime

    torch_version = getattr(torch_module, "__version__", None)
    if torch_version is None:
        version_identity = torch_receipt.get("distribution_version", {})
        if (
            isinstance(version_identity, dict)
            and version_identity.get("status") == "available"
        ):
            torch_version = version_identity.get("value")
    version_module = getattr(torch_module, "version", None)
    cuda_module = getattr(torch_module, "cuda", None)
    cudnn_module = getattr(getattr(torch_module, "backends", None), "cudnn", None)
    cuda_available = _safe_bool_call(cuda_module, "is_available")
    cuda_compiled_version = _string_or_none(getattr(version_module, "cuda", None))
    cudnn_version = _safe_int_call(cudnn_module, "version")
    driver_version = _nvidia_driver_version()
    runtime["torch_cuda"] = {
        "torch_version": _string_or_none(torch_version),
        "torch_git_version": _string_or_none(
            getattr(version_module, "git_version", None)
        ),
        "cuda_compiled_version": cuda_compiled_version,
        "hip_compiled_version": _string_or_none(getattr(version_module, "hip", None)),
        "cuda_available": cuda_available,
        "cuda_device_count": _safe_int_call(cuda_module, "device_count"),
        "cudnn_version": cudnn_version,
        "nvidia_driver_version": driver_version,
        "cuda_driver_runtime_applicable": (
            cuda_available is True
            and cuda_compiled_version is not None
            and cudnn_version is not None
            and driver_version.get("status") == "available"
        ),
    }
    return runtime


def _run_git(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        timeout=10,
    )


def _git_commit(git_root: Path) -> dict[str, Any]:
    try:
        result = _run_git(["rev-parse", "--verify", "HEAD"], cwd=git_root)
    except (
        FileNotFoundError,
        subprocess.TimeoutExpired,
        OSError,
        subprocess.SubprocessError,
    ):
        return _unavailable("commit_unavailable")
    value = result.stdout.strip().decode("ascii", "ignore")
    if (
        result.returncode != 0
        or len(value) not in {40, 64}
        or any(char not in "0123456789abcdef" for char in value)
    ):
        return _unavailable("commit_unavailable")
    return _available(value)


def _parse_git_status(output: bytes) -> list[tuple[str, str]] | None:
    records: list[tuple[str, str]] = []
    for raw_record in output.split(b"\0"):
        if not raw_record:
            continue
        if len(raw_record) < 4 or raw_record[2:3] != b" ":
            return None
        status = raw_record[:2].decode("ascii", "replace")
        path = raw_record[3:].decode("utf-8", "surrogateescape")
        pure_path = PurePosixPath(path)
        if pure_path.is_absolute() or ".." in pure_path.parts or not pure_path.parts:
            return None
        records.append((status, path))
    return records


def _execution_path_class(path: str) -> str | None:
    pure_path = PurePosixPath(path)
    first = pure_path.parts[0]
    if first == "src":
        return "source"
    if first == "configs":
        return "config"
    if first == "scripts":
        return "script"
    if len(pure_path.parts) == 1:
        name = pure_path.name
        if name.endswith(".py"):
            return "source"
        if (
            name in _DEPENDENCY_FILES
            or name.startswith("requirements")
            and name.endswith((".txt", ".in"))
        ):
            return "dependency"
    return None


def _digest_relevant_changes(
    git_root: Path, records: list[tuple[str, str, str]]
) -> dict[str, Any]:
    digest = hashlib.sha256()
    total_bytes = 0
    for status, relative_path, path_class in records:
        path = git_root / relative_path
        identity, consumed_bytes = _worktree_path_identity(
            path, status=status, max_bytes=_MAX_CHANGED_FILE_BYTES
        )
        total_bytes += consumed_bytes
        if total_bytes > _MAX_CHANGED_TOTAL_BYTES:
            return _unavailable("changed_content_too_large")
        if identity["status"] != "available":
            return _unavailable("changed_content_unavailable")
        canonical = json.dumps(
            {
                "content": identity["value"],
                "path": relative_path,
                "path_class": path_class,
                "status": status,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8", "surrogatepass")
        digest.update(len(canonical).to_bytes(8, "big"))
        digest.update(canonical)
    return _available(digest.hexdigest())


def _worktree_path_identity(
    path: Path, *, status: str, max_bytes: int
) -> tuple[dict[str, Any], int]:
    try:
        before = path.lstat()
    except FileNotFoundError:
        if "D" in status:
            return _available("deleted"), 0
        return _unavailable("path_missing"), 0
    except OSError:
        return _unavailable("path_unreadable"), 0
    if stat.S_ISLNK(before.st_mode):
        try:
            target = os.readlink(path).encode("utf-8", "surrogateescape")
        except OSError:
            return _unavailable("symlink_unreadable"), 0
        if len(target) > max_bytes:
            return _unavailable("changed_content_too_large"), len(target)
        return _available("symlink:" + hashlib.sha256(target).hexdigest()), len(target)
    if not stat.S_ISREG(before.st_mode):
        return _unavailable("unsupported_path_type"), 0
    file_digest = _sha256_path(path, max_bytes=max_bytes)
    if file_digest["status"] != "available":
        return file_digest, min(before.st_size, max_bytes + 1)
    try:
        after = path.stat()
    except OSError:
        return _unavailable("changed_during_hash"), before.st_size
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        return _unavailable("changed_during_hash"), before.st_size
    mode = stat.S_IMODE(before.st_mode)
    return _available(f"file:{mode:o}:{file_digest['value']}"), before.st_size


def _sha256_path(path: Path, *, max_bytes: int) -> dict[str, Any]:
    try:
        file_stat = path.stat()
    except FileNotFoundError:
        return _unavailable("origin_missing")
    except OSError:
        return _unavailable("origin_unreadable")
    if not stat.S_ISREG(file_stat.st_mode):
        return _unavailable("origin_not_regular_file")
    if file_stat.st_size > max_bytes:
        return _unavailable("origin_too_large")
    digest = hashlib.sha256()
    consumed = 0
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(
                min(_HASH_CHUNK_BYTES, max_bytes - consumed + 1)
            ):
                consumed += len(chunk)
                if consumed > max_bytes:
                    return _unavailable("origin_too_large")
                digest.update(chunk)
    except OSError:
        return _unavailable("origin_unreadable")
    try:
        after = path.stat()
    except OSError:
        return _unavailable("origin_changed_during_hash")
    if (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_size,
        file_stat.st_mtime_ns,
    ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        return _unavailable("origin_changed_during_hash")
    return _available(digest.hexdigest())


def _path_size(path: Path, *, max_bytes: int) -> dict[str, Any]:
    try:
        file_stat = path.stat()
    except FileNotFoundError:
        return _unavailable("origin_missing")
    except OSError:
        return _unavailable("origin_unreadable")
    if not stat.S_ISREG(file_stat.st_mode):
        return _unavailable("origin_not_regular_file")
    if file_stat.st_size > max_bytes:
        return _unavailable("origin_too_large")
    return _available(file_stat.st_size)


def _nvidia_driver_version() -> dict[str, Any]:
    proc_version = Path("/proc/driver/nvidia/version")
    try:
        text = proc_version.read_text(encoding="utf-8")
    except OSError:
        text = ""
    if text:
        match = re.search(r"Kernel Module\s+([0-9]+(?:\.[0-9]+)+)", text)
        if match is not None:
            return _available(match.group(1))

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (
        FileNotFoundError,
        subprocess.TimeoutExpired,
        OSError,
        subprocess.SubprocessError,
    ):
        return _unavailable("nvidia_driver_unavailable")
    if result.returncode != 0 or len(result.stdout) > _MAX_DRIVER_OUTPUT_BYTES:
        return _unavailable("nvidia_driver_unavailable")
    try:
        values = {
            line.strip()
            for line in result.stdout.decode("ascii", "strict").splitlines()
            if line.strip()
        }
    except UnicodeDecodeError:
        return _unavailable("nvidia_driver_unavailable")
    if len(values) != 1:
        return _unavailable("nvidia_driver_unavailable")
    value = values.pop()
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", value) is None:
        return _unavailable("nvidia_driver_unavailable")
    return _available(value)


def _resolve_distribution_version(distribution: str) -> dict[str, Any]:
    try:
        value = _distribution_version(distribution)
    except PackageNotFoundError:
        return _unavailable("distribution_not_found")
    except Exception:
        return _unavailable("version_resolution_failed")
    if not isinstance(value, str) or not value:
        return _unavailable("version_resolution_failed")
    return _available(value)


def _resolve_distribution_record(distribution: str) -> dict[str, Any]:
    try:
        text = _distribution_record_text(distribution)
    except PackageNotFoundError:
        return _unavailable("distribution_not_found")
    except Exception:
        return _unavailable("distribution_record_resolution_failed")
    if text is None:
        return _unavailable("distribution_record_absent")
    if not isinstance(text, str):
        return _unavailable("distribution_record_invalid")
    encoded = text.encode("utf-8")
    if len(encoded) > _MAX_DISTRIBUTION_RECORD_BYTES:
        return _unavailable("distribution_record_too_large")
    return _available(
        {
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "size_bytes": len(encoded),
        }
    )


def _source_repository_identity(origin: Path) -> dict[str, Any]:
    try:
        probe = _run_git(["rev-parse", "--show-toplevel"], cwd=origin.parent)
    except (
        FileNotFoundError,
        subprocess.TimeoutExpired,
        OSError,
        subprocess.SubprocessError,
    ):
        return _unavailable("source_repository_unavailable")
    if probe.returncode != 0 or len(probe.stdout) > _MAX_GIT_OUTPUT_BYTES:
        return _unavailable("source_repository_unavailable")
    try:
        root = Path(os.fsdecode(probe.stdout.strip())).resolve(strict=True)
    except (OSError, ValueError):
        return _unavailable("source_repository_unavailable")
    commit = _git_commit(root)
    if commit.get("status") != "available":
        return _unavailable("source_repository_commit_unavailable")
    try:
        relative_origin = origin.relative_to(root)
    except ValueError:
        return _unavailable("source_repository_origin_outside_root")
    if not relative_origin.parts:
        return _unavailable("source_repository_origin_unavailable")
    package_root = relative_origin.parts[0]
    pathspecs = [package_root, *sorted(_DEPENDENCY_FILES)]
    try:
        status_result = _run_git(
            [
                "status",
                "--porcelain=v1",
                "-z",
                "--untracked-files=all",
                "--no-renames",
                "--",
                *pathspecs,
            ],
            cwd=root,
        )
    except (
        FileNotFoundError,
        subprocess.TimeoutExpired,
        OSError,
        subprocess.SubprocessError,
    ):
        return _unavailable("source_repository_status_unavailable")
    if (
        status_result.returncode != 0
        or len(status_result.stdout) > _MAX_GIT_OUTPUT_BYTES
    ):
        return _unavailable("source_repository_status_unavailable")
    records = _parse_git_status(status_result.stdout)
    if records is None:
        return _unavailable("source_repository_status_parse_failed")
    if not records:
        return _available({"commit": commit["value"], "state": "clean"})
    relevant = [(status, path, "dependency_source") for status, path in records]
    if len(relevant) > _MAX_CHANGE_RECORDS:
        return _unavailable("too_many_dependency_source_changes")
    digest = _digest_relevant_changes(root, relevant)
    if digest.get("status") != "available":
        return _unavailable("dependency_source_changes_unavailable")
    return _available(
        {
            "commit": commit["value"],
            "state": "dirty",
            "local_changes_digest": digest["value"],
            "changed_path_count": len(relevant),
        }
    )


def _module_origin(module: object) -> Path | None:
    spec = getattr(module, "__spec__", None)
    value = getattr(spec, "origin", None) or getattr(module, "__file__", None)
    return _resolved_origin(value)


def _reference_module_origin(import_name: str) -> Path | None:
    try:
        spec = _find_module_spec(import_name)
    except Exception:
        return None
    return _resolved_origin(getattr(spec, "origin", None))


def _resolved_origin(value: object) -> Path | None:
    if not isinstance(value, str) or not value or value in {"built-in", "frozen"}:
        return None
    try:
        return Path(value).resolve(strict=False)
    except (OSError, ValueError):
        return None


def _origin_kind(origin: Path) -> str:
    binary_suffixes = {".so", ".pyd", ".dll", ".dylib"}
    name = origin.name.lower()
    return (
        "binary"
        if any(
            name.endswith(suffix) or f"{suffix}." in name for suffix in binary_suffixes
        )
        else "source"
    )


def _import_module(name: str) -> object:
    return importlib.import_module(name)


def _find_module_spec(name: str) -> object:
    return _stdlib_find_spec(name)


def _distribution_version(distribution: str) -> str:
    return _metadata_version(distribution)


def _distribution_record_text(distribution: str) -> str | None:
    return _metadata_distribution(distribution).read_text("RECORD")


def _distribution_file_path(distribution: str, relative_path: str) -> Path | None:
    try:
        value = _metadata_distribution(distribution).locate_file(relative_path)
        return Path(value).resolve(strict=False)
    except Exception:
        return None


def _loaded_shared_object_origin(soname: str) -> Path | None:
    """Resolve one actually mapped shared object without invoking a loader."""

    try:
        lines = Path("/proc/self/maps").read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return None
    origins: set[Path] = set()
    for line in lines:
        fields = line.split(maxsplit=5)
        if len(fields) != 6:
            continue
        raw_path = fields[5]
        if raw_path.endswith(" (deleted)"):
            continue
        path = Path(raw_path)
        if not path.is_absolute() or path.name != soname:
            continue
        try:
            resolved = path.resolve(strict=True)
            file_stat = resolved.stat()
        except (OSError, RuntimeError):
            continue
        if stat.S_ISREG(file_stat.st_mode):
            origins.add(resolved)
    if len(origins) != 1:
        return None
    return origins.pop()


def _mapped_shared_object_origins() -> dict[str, tuple[Path, ...]]:
    try:
        lines = Path("/proc/self/maps").read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return {}
    origins: dict[str, set[Path]] = {}
    for line in lines:
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or fields[5].endswith(" (deleted)"):
            continue
        path = Path(fields[5])
        name = path.name
        relevant = (
            name in {"libtorch_cuda.so", "libc10_cuda.so", "libcudart.so.12"}
            or name.startswith("libcuda.so")
            or name.startswith("libcudnn")
            and ".so.9" in name
        )
        if not relevant or not path.is_absolute():
            continue
        try:
            resolved = path.resolve(strict=True)
            file_stat = resolved.stat()
        except (OSError, RuntimeError):
            continue
        if stat.S_ISREG(file_stat.st_mode):
            origins.setdefault(name, set()).add(resolved)
    return {
        name: tuple(sorted(paths, key=str)) for name, paths in sorted(origins.items())
    }


def _cuda_runtime_is_initialized() -> bool:
    try:
        torch_module = _import_module("torch")
    except Exception:
        return False
    cuda_module = getattr(torch_module, "cuda", None)
    return _safe_bool_call(cuda_module, "is_initialized") is True


def _elf_build_id(path: Path) -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["readelf", "-n", str(path)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except (
        FileNotFoundError,
        subprocess.TimeoutExpired,
        OSError,
        subprocess.SubprocessError,
    ):
        return _unavailable("elf_build_id_unavailable")
    if result.returncode != 0 or len(result.stdout) > _MAX_READELF_OUTPUT_BYTES:
        return _unavailable("elf_build_id_unavailable")
    try:
        text = result.stdout.decode("ascii", "strict")
    except UnicodeDecodeError:
        return _unavailable("elf_build_id_unavailable")
    matches = re.findall(r"Build ID:\s*([0-9a-fA-F]+)", text)
    if len(matches) != 1:
        return _unavailable("elf_build_id_unavailable")
    return _available(matches[0].lower())


def _safe_bool_call(owner: object, name: str) -> bool | None:
    value = _safe_call(owner, name)
    return value if isinstance(value, bool) else None


def _safe_int_call(owner: object, name: str) -> int | None:
    value = _safe_call(owner, name)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _safe_call(owner: object, name: str) -> object:
    function = getattr(owner, name, None)
    if not callable(function):
        return None
    try:
        return function()
    except Exception:
        return None


def _string_or_none(value: object) -> str | None:
    return None if value is None else str(value)


def _available(value: object) -> dict[str, Any]:
    return {"status": "available", "value": value}


def _unavailable(reason: str) -> dict[str, str]:
    return {"status": "unavailable", "reason": reason}


def _empty_change_metadata() -> dict[str, Any]:
    return {"count": 0, "path_classes": {}, "truncated": False}


def _unavailable_repository(reason: str) -> dict[str, Any]:
    return {
        "commit": _unavailable(reason),
        "state": "unavailable",
        "tracked_changes_present": None,
        "untracked_changes_present": None,
        "execution_relevant_changes": _empty_change_metadata(),
        "execution_relevant_digest": _unavailable(reason),
    }
