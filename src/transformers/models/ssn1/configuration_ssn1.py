# coding=utf-8
# Copyright 2024 The SSN1 Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""SSN1 model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import RotaryEmbeddingConfigMixin
from transformers.utils import logging


logger = logging.get_logger(__name__)


class SSN1Config(PretrainedConfig, RotaryEmbeddingConfigMixin):
    r"""
    This is the configuration class to store the configuration of a [`SSN1Model`]. It is used to instantiate a SSN1
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the SSN1-1B architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 128256):
            Vocabulary size of the SSN1 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`SSN1Model`]
        hidden_size (`int`, *optional*, defaults to 2048):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 7168):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 16):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used.
        head_dim (`int`, *optional*, defaults to 64):
            The attention head dimension.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with.
        rope_theta (`float`, *optional*, defaults to 500000.0):
            The base period of the RoPE embeddings.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        use_qk_norm (`bool`, *optional*, defaults to `True`):
            Whether to use normalization on Q and K projections before applying RoPE.
        norm_reorder (`bool`, *optional*, defaults to `False`):
            Whether to use post-norm (True) or pre-norm (False). Pre-norm applies normalization before
            attention/FFN, post-norm applies it after.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the MLP layers.

    Example:
    ```python
    >>> from transformers import SSN1Model, SSN1Config

    >>> # Initializing a SSN1 style configuration
    >>> configuration = SSN1Config()

    >>> # Initializing a model from the configuration
    >>> model = SSN1Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
```"""

    model_type = "ssn1"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=128256,
        hidden_size=2048,
        intermediate_size=7168,
        num_hidden_layers=16,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=64,
        max_position_embeddings=2048,
        rope_theta=500000.0,
        rms_norm_eps=1e-5,
        use_qk_norm=True,
        norm_reorder=False,
        attention_bias=False,
        mlp_bias=False,
        attention_dropout=0.0,
        hidden_act="silu",
        initializer_range=0.02,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=128000,
        eos_token_id=128001,
        tie_word_embeddings=True,
        rope_scaling=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.use_qk_norm = use_qk_norm
        self.norm_reorder = norm_reorder
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias
        self.attention_dropout = attention_dropout
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.use_cache = use_cache

        # Rope parameters handling (following Llama pattern)
        if rope_scaling is not None:
            self.rope_parameters = rope_scaling
        else:
            self.rope_parameters = {"rope_type": "default", "rope_theta": self.rope_theta}

        # Set special token ids before calling super().__init__
        self.tie_word_embeddings = tie_word_embeddings
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        super().__init__(**kwargs)