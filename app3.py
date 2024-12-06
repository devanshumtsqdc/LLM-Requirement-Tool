import streamlit as st
import json
from config import DATA_TYPES, PARAMETERS, OPTIMIZERS
from utils import calculate_inference_memory, calculate_training_memory

# ----------------- Streamlit Setup ----------------- #
st.set_page_config(page_title="LLM Memory Requirements")
st.title("LLM Memory Requirements")

# ----------------- Sidebar Initialization ----------------- #
# Initialize session state for uploaded models
if "uploaded_models" not in st.session_state:
    st.session_state.uploaded_models = {}  # Stores uploaded models as {name: content}

def set_values():
    """Update the values based on the selected model."""
    if st.session_state.model in st.session_state.uploaded_models:
        model_info = st.session_state.uploaded_models[st.session_state.model]
        for param in PARAMETERS:
            st.session_state[param] = model_info.get(PARAMETERS[param], None)
    else:
        for param in PARAMETERS:
            st.session_state[param] = None

# Function to handle file uploads
def handle_file_upload(uploaded_file):
    if uploaded_file:
        # Read file content and save in session state
        try:
            content = json.load(uploaded_file)
            st.session_state.uploaded_models[uploaded_file.name] = content
            st.success(f"Model '{uploaded_file.name}' uploaded successfully!")
        except json.JSONDecodeError:
            st.error("Failed to parse JSON. Please upload a valid model file.")

# ----------------- Sidebar UI ----------------- #
# File upload section
uploaded_file = st.sidebar.file_uploader("Upload Model", type=["json"], key="file_uploader")
if uploaded_file:
    handle_file_upload(uploaded_file)

# Model Selection
model_keys = list(st.session_state.uploaded_models.keys())
model = st.sidebar.selectbox(
    "Model", model_keys, index=0 if model_keys else None, on_change=set_values, key="model"
)

# Parameters
model_size = st.sidebar.number_input(
    "Number of parameters (in billions)",
    min_value=0,
    step=1,
    value=st.session_state.get("model_size", None),
    key="model_size",
    help="Number of parameters in the model in billions",
)
precision = st.sidebar.selectbox(
    "Precision",
    DATA_TYPES,
    index=None,
    key="precision",
    help="Data type used (int 8 and int 4 are for quantization)",
)
batch_size = st.sidebar.number_input(
    "Batch Size",
    min_value=0,
    step=1,
    value=st.session_state.get("batch_size", 1),
    key="batch_size",
)
sequence_length = st.sidebar.number_input(
    "Sequence Length",
    min_value=0,
    step=1,
    value=st.session_state.get("sequence_length", 2048),
    key="sequence_length",
    help="Number of tokens in the input sequence.",
)
hidden_size = st.sidebar.number_input(
    "Hidden Size",
    min_value=0,
    step=1,
    value=st.session_state.get("hidden_size", None),
    key="hidden_size",
    help="Size of the hidden layer (given by the model card).",
)
num_hidden_layers = st.sidebar.number_input(
    "Number of Layers",
    min_value=0,
    step=1,
    value=st.session_state.get("num_hidden_layers", None),
    key="num_hidden_layers",
    help="Number of layers in the model (given by the model card).",
)
num_attention_heads = st.sidebar.number_input(
    "Number of Attention Heads",
    min_value=0,
    step=1,
    value=st.session_state.get("num_attention_heads", None),
    key="num_attention_heads",
    help="Number of attention heads in the model (given by the model card).",
)


# ----------------- Main Screen UI ----------------- #
# Dividing the screen into two tabs
inference, training = st.tabs(["Inference", "Training"])

# Tab 2: Training
training1, training2 = training.columns(2)
optimizer = training2.selectbox("Optimizer", list(OPTIMIZERS.keys()), key="optimizer")
trainable_parameters = training2.slider(
    "Percentage of trainable parameters", 0, 100, 100, key="trainable_params"
)

# Inference Memory
inference_memory = calculate_inference_memory(
    model_size,
    precision,
    batch_size,
    sequence_length,
    hidden_size,
    num_hidden_layers,
    num_attention_heads,
)

inference.write(f"**Total Inference Memory**: {inference_memory['inference_memory']}")
inference.write(f"- **Model Weights**: {inference_memory['model_weights']}")
inference.write(f"- **KV Cache**: {inference_memory['kv_cache']}")
inference.write(f"- **Activation Memory**: {inference_memory['activation_memory']}")

# Training Memory
training_memory = calculate_training_memory(
    model_size,
    precision,
    batch_size,
    sequence_length,
    hidden_size,
    num_hidden_layers,
    num_attention_heads,
    optimizer,
    trainable_parameters,
)

training1.write(f"**Total Training Memory**: {training_memory['training_memory']}")
training1.write(f"- **Model Weights**: {training_memory['model_weights']}")
training1.write(f"- **KV Cache**: {training_memory['kv_cache']}")
training1.write(f"- **Activation Memory**: {training_memory['activation_memory']}")
training1.write(f"- **Optimizer Memory**: {training_memory['optimizer_memory']}")
training1.write(f"- **Gradients Memory**: {training_memory['gradients_memory']}")

# ----------------- Error Handling ----------------- #
if None in st.session_state.values():
    st.warning("Some information is missing.")
