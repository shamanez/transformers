from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.ssn1 import SSN1Config, SSN1ForCausalLM

# Create from config
config = SSN1Config(
    vocab_size=32000,
    hidden_size=2048,
    intermediate_size=7168,
    num_hidden_layers=16,
    num_attention_heads=32,
    num_key_value_heads=8,
    use_qk_norm=True,
    norm_reorder=False,
)
model = SSN1ForCausalLM(config)

print(model)

# Save and reload with Auto classes
# model.save_pretrained("./my_ssn1_model")
# loaded_model = AutoModelForCausalLM.from_pretrained("./my_ssn1_model")
