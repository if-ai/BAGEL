import os
import sys
import torch
import numpy as np
import random
import subprocess
from typing import Dict, Tuple, Optional, Any, Union
from PIL import Image
from folder_paths import folder_names_and_paths
# import comfy.utils # For progress bar - Temporarily commented out
import bitsandbytes as bnb
from torch import nn

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import BAGEL related modules
try:
    from accelerate import (
        infer_auto_device_map,
        load_checkpoint_and_dispatch,
        init_empty_weights,
    )
    from data.data_utils import add_special_tokens, pil_img2rgb
    from data.transforms import ImageTransform
    from inferencer import InterleaveInferencer
    from modeling.autoencoder import load_ae
    from modeling.bagel.qwen2_navit import NaiveCache
    from modeling.bagel import (
        BagelConfig,
        Bagel,
        Qwen2Config,
        Qwen2ForCausalLM,
        SiglipVisionConfig,
        SiglipVisionModel,
    )
    from modeling.qwen2 import Qwen2Tokenizer
except ImportError as e:
    print(f"Error importing BAGEL modules: {e}")
    print("Please ensure BAGEL model files are properly installed.")

# Register the BAGEL model folder
models_dir = os.path.join(os.getcwd(), "models")
folder_names_and_paths["bagel"] = (
    [os.path.join(models_dir, "bagel")],
    [".json", ".safetensors"],
)


def set_seed(seed: int) -> int:
    """Set random seeds for reproducibility"""
    if seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


def download_model_with_git(
    model_dir: str, repo_id: str = "ByteDance-Seed/BAGEL-7B-MoT"
) -> str:
    """
    Download model using git lfs (recommended method)

    Args:
        model_dir: Directory to download the repo to (repo files will be placed directly here)
        repo_id: Hugging Face repository ID

    Returns:
        Path to the downloaded model if successful, None otherwise
    """
    try:
        print(f"Downloading BAGEL model using git lfs to {model_dir}...")

        # Create parent directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)

        # Check if git lfs is installed
        try:
            subprocess.run(["git", "lfs", "version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Git LFS not found. Installing git lfs...")
            subprocess.run(["git", "lfs", "install"], check=True)

        # Clone the repository directly to model_dir
        clone_cmd = ["git", "clone", f"https://huggingface.co/{repo_id}", model_dir]

        result = subprocess.run(clone_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Successfully downloaded BAGEL model to {model_dir}")
            return model_dir
        else:
            print(f"Git clone failed: {result.stderr}")
            return None

    except Exception as e:
        print(f"Error downloading model with git: {e}")
        return None


def download_model_with_hf_hub(
    model_dir: str, repo_id: str = "ByteDance-Seed/BAGEL-7B-MoT"
) -> str:
    """
    Download model using huggingface_hub (fallback method)

    Args:
        model_dir: Directory to download the repo to (repo files will be placed directly here)
        repo_id: Hugging Face repository ID

    Returns:
        Path to the downloaded model if successful, None otherwise
    """
    try:
        from huggingface_hub import snapshot_download

        print(f"Downloading BAGEL model using huggingface_hub to {model_dir}...")

        # Create parent directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)

        # Download the entire repository directly to model_dir
        snapshot_download(
            repo_id=repo_id, local_dir=model_dir, local_dir_use_symlinks=False
        )

        print(f"Successfully downloaded BAGEL model to {model_dir}")
        return model_dir

    except ImportError:
        print(
            "huggingface_hub not installed. Please install it with: pip install huggingface_hub"
        )
        return None
    except Exception as e:
        print(f"Error downloading model with huggingface_hub: {e}")
        return None


def check_model_files(model_path: str, precision: str = "bfloat16") -> bool:
    """
    Check if all required model files exist for the given precision.

    Args:
        model_path: Path to the model directory
        precision: The precision string (e.g., "bfloat16", "fp8_e4m3fn")

    Returns:
        True if all files exist, False otherwise
    """
    required_files = [
        "llm_config.json",
        "vit_config.json",
        "ae.safetensors",
    ]
    expected_ema_file = "ema.safetensors"
    if precision.startswith("fp8"):
        expected_ema_file = "ema-FP8.safetensors"
    elif precision == "int8":
        expected_ema_file = "model_int8.safetensors"
    
    required_files.append(expected_ema_file)

    for file_name in required_files:
        if not os.path.exists(os.path.join(model_path, file_name)):
            print(f"Error: Missing required file for precision '{precision}': {file_name} in {model_path}")
            return False
    return True


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert PIL image to ComfyUI tensor format"""
    img_array = np.array(img).astype(np.float32) / 255.0
    if len(img_array.shape) == 3:
        img_tensor = torch.from_numpy(img_array)[None,]  # Add batch dimension
    else:
        img_tensor = torch.from_numpy(img_array)
    return img_tensor


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert ComfyUI tensor to PIL image"""
    if len(tensor.shape) == 4:
        tensor = tensor[0]  # Remove batch dimension
    img_array = (tensor.cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(img_array)


def quantize_linear_layer(layer: nn.Linear):
    ql = bnb.nn.Linear8bitLt(
        layer.in_features,
        layer.out_features,
        bias=layer.bias is not None,
        has_fp16_weights=True   # weights kept fp16 in memory, multiplied in int8
    )
    ql.weight.data = layer.weight.data  # copy weights
    if layer.bias is not None:
        ql.bias.data = layer.bias.data
    return ql


class BagelModelLoader:
    """BAGEL Model Loader Node"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": (
                    "STRING",
                    {
                        "default": "ByteDance-Seed/BAGEL-7B-MoT",
                        "tooltip": "Hugging Face model repo ID or local path. For FP8 models, ensure this path points to the FP8 version (e.g., 'meimeilook/BAGEL-7B-MoT-FP8') and select the corresponding FP8 precision. For INT8, ensure it points to an INT8 .safetensors model.",
                    },
                ),
                "precision": (
                    ["bfloat16", "fp8_e4m3fn", "fp8_e5m2 (coming soon)", "int8"],
                    {"default": "bfloat16", "tooltip": "Model precision. Select fp8 for FP8 quantized models. Select int8 for INT8 quantized .safetensors models."},
                ),
            }
        }

    RETURN_TYPES = ("BAGEL_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "BAGEL/Core"

    @classmethod
    def VALIDATE_INPUTS(cls, model_path, precision):
        """Validate input parameters"""
        if not isinstance(model_path, str) or not model_path.strip():
            return "Model path must be a non-empty string"
        if precision not in ["bfloat16", "fp8_e4m3fn", "fp8_e5m2 (coming soon)", "int8"]:
            return "Invalid precision selected."
        if precision == "fp8_e5m2 (coming soon)":
            print("Note: fp8_e5m2 precision is marked as 'coming soon' and is experimental.")
        if precision == "int8":
            print("Note: INT8 precision assumes a .safetensors model with INT8 weights (e.g., model_int8.safetensors) but will be loaded as bfloat16.")
        return True

    def load_model(self, model_path: str, precision: str) -> Tuple[Dict[str, Any]]:
        """
        Load BAGEL model and its components. Automatically download the model if not found.

        Args:
            model_path: URL to the Hugging Face model repository or local path
            precision: The precision to load the model with ("bfloat16", "fp8_e4m3fn", "fp8_e5m2 (coming soon)", "int8")

        Returns:
            Dictionary containing all model components
        """
        try:
            # pbar = comfy.utils.ProgressBar(10) # Temporarily commented out
            # current_step = 0

            # def update_pbar(step_increment=1, description=""):
            #     nonlocal current_step
            #     current_step += step_increment
            #     pbar.update_absolute(current_step, 10, description if description else None)

            base_model_dir = os.path.join(os.getcwd(), "models", "bagel")
            if os.path.isabs(model_path) and os.path.exists(model_path):
                local_model_dir = model_path
            else:
                repo_name = model_path.split("/")[-1]
                local_model_dir = os.path.join(base_model_dir, repo_name)
            # update_pbar(description="Checking local files...") # Temporarily commented out

            if not os.path.exists(local_model_dir) or not check_model_files(local_model_dir, precision):
                print(
                    f"Model not found locally or missing files for precision '{precision}'. Attempting to download from {model_path}..."
                )
                downloaded_path = download_model_with_hf_hub(
                    local_model_dir, repo_id=model_path
                )
                if not downloaded_path:
                    raise FileNotFoundError(
                        f"Failed to download BAGEL model from {model_path}. "
                        f"Please manually download it and place it in {local_model_dir}"
                    )
                print(f"Successfully downloaded BAGEL model to {local_model_dir}")
                # update_pbar(description="Download complete.") # Temporarily commented out
            
            # update_pbar(description=f"Loading {precision} model...") # Temporarily commented out

            # Final check that all required files exist for the specified precision
            if not check_model_files(local_model_dir, precision):
                raise FileNotFoundError(
                    f"Required model files missing in {local_model_dir} for precision '{precision}'. "
                    f"Please ensure the correct model version and all its files are present."
                )
            
            print(f"Loading model with precision: {precision}")

            # Load configuration files
            llm_config = Qwen2Config.from_json_file(
                os.path.join(local_model_dir, "llm_config.json")
            )
            llm_config.qk_norm = True
            llm_config.tie_word_embeddings = False
            llm_config.layer_module = "Qwen2MoTDecoderLayer"

            vit_config = SiglipVisionConfig.from_json_file(
                os.path.join(local_model_dir, "vit_config.json")
            )
            vit_config.rope = False
            vit_config.num_hidden_layers -= 1

            vae_model, vae_config = load_ae(
                local_path=os.path.join(local_model_dir, "ae.safetensors")
            )

            # Create BAGEL configuration
            config = BagelConfig(
                visual_gen=True,
                visual_und=True,
                llm_config=llm_config,
                vit_config=vit_config,
                vae_config=vae_config,
                vit_max_num_patch_per_side=70,
                connector_act="gelu_pytorch_tanh",
                latent_patch_size=2,
                max_latent_size=64,
            )

            # Initialize model
            with init_empty_weights():
                language_model = Qwen2ForCausalLM(llm_config)
                vit_model = SiglipVisionModel(vit_config)
                model = Bagel(language_model, vit_model, config)
                model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(
                    vit_config, meta=True
                )

            # Load tokenizer
            tokenizer = Qwen2Tokenizer.from_pretrained(local_model_dir)
            tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

            # Create transformers
            vae_transform = ImageTransform(1024, 512, 16)
            vit_transform = ImageTransform(980, 224, 14)
            
            # Determine checkpoint file and dtype for loading
            checkpoint_to_load = "ema.safetensors"
            torch_dtype_for_load = torch.bfloat16
            if precision.startswith("fp8"):
                checkpoint_to_load = "ema-FP8.safetensors"
                torch_dtype_for_load = torch.bfloat16 
                print(f"Using FP8 checkpoint: {checkpoint_to_load}. Weights will be loaded as {torch_dtype_for_load} (FP8 on disk).")
            elif precision == "int8":
                checkpoint_to_load = "model_int8.safetensors"
                torch_dtype_for_load = torch.bfloat16 
                print(f"Using INT8 checkpoint: {checkpoint_to_load}. Weights will be loaded as {torch_dtype_for_load} (INT8 on disk).")
            else:
                 print(f"Using bfloat16 checkpoint: {checkpoint_to_load}. Weights will be loaded as {torch_dtype_for_load}.")
            
            full_checkpoint_path = os.path.join(local_model_dir, checkpoint_to_load)
            # update_pbar(description=f"Loading {checkpoint_to_load}...") # Temporarily commented out

            if not os.path.exists(full_checkpoint_path):
                # This should ideally be caught by check_model_files earlier
                raise FileNotFoundError(f"Checkpoint file {full_checkpoint_path} not found for precision {precision}.")

            # Load model weights
            model = load_checkpoint_and_dispatch(
                model,
                checkpoint=full_checkpoint_path,
                device_map="auto",
                dtype=torch_dtype_for_load,
                force_hooks=True,
            ).eval()
            # update_pbar(description="Model weights loaded.") # Temporarily commented out

            # Create inferencer
            inferencer = InterleaveInferencer(
                model=model,
                vae_model=vae_model,
                tokenizer=tokenizer,
                vae_transform=vae_transform,
                vit_transform=vit_transform,
                new_token_ids=new_token_ids,
            )
            # update_pbar(description="Inferencer created.") # Temporarily commented out

            # Wrap as model dictionary
            model_dict = {
                "model": model,
                "inferencer": inferencer,
                "tokenizer": tokenizer,
                "vae_model": vae_model,
                "vae_transform": vae_transform,
                "vit_transform": vit_transform,
                "config": config,
                "model_path": local_model_dir,
            }
            # pbar.update_absolute(10, 10, "Model loaded!") # Temporarily commented out
            print(f"Successfully loaded BAGEL model from {local_model_dir}")
            return (model_dict,)

        except Exception as e:
            print(f"Error loading BAGEL model: {e}")
            raise e


class BagelTextToImage:
    """BAGEL Text to Image Node"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("BAGEL_MODEL", {"tooltip": "BAGEL model"}),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "A female cosplayer portraying an ethereal fairy or elf, wearing a flowing dress made of delicate fabrics in soft, mystical colors like emerald green and silver. She has pointed ears, a gentle, enchanting expression, and her outfit is adorned with sparkling jewels and intricate patterns. The background is a magical forest with glowing plants, mystical creatures, and a serene atmosphere.",
                        "tooltip": "Text prompt",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 1000000,
                        "tooltip": "Random seed, 0 for random",
                    },
                ),
                "image_ratio": (
                    ["1:1", "4:3", "3:4", "16:9", "9:16"],
                    {"default": "1:1", "tooltip": "Image aspect ratio"},
                ),
                "cfg_text_scale": (
                    "FLOAT",
                    {
                        "default": 4.0,
                        "min": 1.0,
                        "max": 8.0,
                        "step": 0.1,
                        "tooltip": "CFG text scaling",
                    },
                ),
                "num_timesteps": (
                    "INT",
                    {
                        "default": 50,
                        "min": 10,
                        "max": 100,
                        "step": 5,
                        "tooltip": "Denoising steps",
                    },
                ),
            },
            "optional": {
                "show_thinking": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Display reasoning process"},
                ),
                "cfg_interval": (
                    "FLOAT",
                    {
                        "default": 0.4,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "tooltip": "CFG interval start value",
                    },
                ),
                "timestep_shift": (
                    "FLOAT",
                    {
                        "default": 3.0,
                        "min": 1.0,
                        "max": 5.0,
                        "step": 0.5,
                        "tooltip": "Timestep offset",
                    },
                ),
                "cfg_renorm_min": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "tooltip": "CFG re-normalization minimum value",
                    },
                ),
                "cfg_renorm_type": (
                    ["global", "local", "text_channel"],
                    {"default": "global", "tooltip": "CFG re-normalization type"},
                ),
                "text_temperature": (
                    "FLOAT",
                    {
                        "default": 0.3,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "tooltip": "Text generation temperature",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "thinking")
    FUNCTION = "generate_image"
    CATEGORY = "BAGEL/Core"

    def generate_image(
        self,
        model: Dict[str, Any],
        prompt: str,
        seed: int,
        image_ratio: str,
        cfg_text_scale: float,
        num_timesteps: int,
        show_thinking: bool = False,
        cfg_interval: float = 0.4,
        timestep_shift: float = 3.0,
        cfg_renorm_min: float = 1.0,
        cfg_renorm_type: str = "global",
        text_temperature: float = 0.3,
    ) -> Tuple[torch.Tensor, str]:
        _prompt_input = prompt
        actual_prompt: str

        if _prompt_input is None: # Handle None input for prompt
            actual_prompt = ""
        elif isinstance(_prompt_input, (list, tuple)):
            if len(_prompt_input) == 1 and isinstance(_prompt_input[0], str):
                actual_prompt = _prompt_input[0]
            else:
                print(f"Error: Invalid prompt format in generate_image: {_prompt_input}. Expected string or list/tuple of one string.")
                empty_image = torch.zeros((1, 512, 512, 3)) 
                return (empty_image, "Error: Invalid prompt format during execution.")
        elif isinstance(_prompt_input, str):
            actual_prompt = _prompt_input
        else:
            print(f"Error: Invalid prompt type in generate_image: {type(_prompt_input)}. Expected string or list/tuple of one string.")
            empty_image = torch.zeros((1, 512, 512, 3))
            return (empty_image, "Error: Invalid prompt type during execution.")

        try:
            set_seed(seed)
            inferencer = model["inferencer"]
            image_shapes_map = {
                "1:1": (1024, 1024),
                "4:3": (768, 1024),
                "3:4": (1024, 768),
                "16:9": (576, 1024),
                "9:16": (1024, 576),
            }
            image_shapes = image_shapes_map[image_ratio]
            inference_hyper = {
                "max_think_token_n": 1024 if show_thinking else 1024,
                "do_sample": False if not show_thinking else False,
                "text_temperature": text_temperature if show_thinking else 0.3,
                "cfg_text_scale": cfg_text_scale,
                "cfg_interval": [cfg_interval, 1.0],
                "timestep_shift": timestep_shift,
                "num_timesteps": num_timesteps,
                "cfg_renorm_min": cfg_renorm_min,
                "cfg_renorm_type": cfg_renorm_type,
                "image_shapes": image_shapes,
            }
            result = inferencer(text=actual_prompt, think=show_thinking, **inference_hyper)
            
            # Convert image format - potentially a list of images
            output_image_or_images = result["image"]

            if isinstance(output_image_or_images, list):
                if not output_image_or_images:
                    print("Warning: Inferencer returned an empty list of images.")
                    empty_image = torch.zeros((1, image_shapes[1], image_shapes[0], 3)) # H, W, C assuming ComfyUI is N H W C
                    return (empty_image, "Error: No images generated.")
                
                tensor_images_list = [pil_to_tensor(img_item) for img_item in output_image_or_images]
                if not tensor_images_list: # Should be caught by the above, but defensive
                     empty_image = torch.zeros((1, image_shapes[1], image_shapes[0], 3))
                     return (empty_image, "Error: Failed to convert images to tensors.")

                final_tensor_image = torch.cat(tensor_images_list, dim=0)
                print(f"Generated image sequence with {len(output_image_or_images)} frames. Final shape: {final_tensor_image.shape}")
            elif isinstance(output_image_or_images, Image.Image):
                final_tensor_image = pil_to_tensor(output_image_or_images)
                print(f"Generated single image with size: {output_image_or_images.size}. Final shape: {final_tensor_image.shape}")
            else:
                print(f"Error: Inferencer returned an unexpected image type: {type(output_image_or_images)}")
                empty_image = torch.zeros((1, image_shapes[1], image_shapes[0], 3))
                return (empty_image, "Error: Unexpected image data from model.")

            # Get reasoning process
            thinking_text = result.get("text", "") if show_thinking else ""

            return (final_tensor_image, thinking_text)

        except Exception as e:
            print(f"Error in text to image generation: {e}")
            # Return empty image and error message
            empty_image = torch.zeros((1, 512, 512, 3))
            return (empty_image, f"Error: {str(e)}")


class BagelImageEdit:
    """BAGEL Image Edit Node"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("BAGEL_MODEL", {"tooltip": "BAGEL model"}),
                "image": ("IMAGE", {"tooltip": "Input image"}),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Edit the image according to the description",
                        "tooltip": "Editing prompt",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 1000000,
                        "tooltip": "Random seed, 0 for random",
                    },
                ),
                "cfg_text_scale": (
                    "FLOAT",
                    {
                        "default": 4.0,
                        "min": 1.0,
                        "max": 8.0,
                        "step": 0.1,
                        "tooltip": "CFG text scaling",
                    },
                ),
                "cfg_img_scale": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 1.0,
                        "max": 4.0,
                        "step": 0.1,
                        "tooltip": "CFG image scaling",
                    },
                ),
                "num_timesteps": (
                    "INT",
                    {
                        "default": 50,
                        "min": 10,
                        "max": 100,
                        "step": 5,
                        "tooltip": "Denoising steps",
                    },
                ),
            },
            "optional": {
                "show_thinking": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Display reasoning process"},
                ),
                "cfg_interval": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "tooltip": "CFG interval start value",
                    },
                ),
                "timestep_shift": (
                    "FLOAT",
                    {
                        "default": 3.0,
                        "min": 1.0,
                        "max": 10.0,
                        "step": 0.5,
                        "tooltip": "Timestep offset",
                    },
                ),
                "cfg_renorm_min": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "tooltip": "CFG re-normalization minimum value",
                    },
                ),
                "cfg_renorm_type": (
                    ["global", "local", "text_channel"],
                    {"default": "text_channel", "tooltip": "CFG re-normalization type"},
                ),
                "text_temperature": (
                    "FLOAT",
                    {
                        "default": 0.3,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "tooltip": "Text generation temperature",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "thinking")
    FUNCTION = "edit_image"
    CATEGORY = "BAGEL/Core"

    def edit_image(
        self,
        model: Dict[str, Any],
        image: torch.Tensor,
        prompt: str,
        seed: int,
        cfg_text_scale: float,
        cfg_img_scale: float,
        num_timesteps: int,
        show_thinking: bool = False,
        cfg_interval: float = 0.0,
        timestep_shift: float = 3.0,
        cfg_renorm_min: float = 1.0,
        cfg_renorm_type: str = "text_channel",
        text_temperature: float = 0.3,
    ) -> Tuple[torch.Tensor, str]:
        _prompt_input = prompt
        actual_prompt: str

        if _prompt_input is None: # Handle None input for prompt
            actual_prompt = ""
        elif isinstance(_prompt_input, (list, tuple)):
            if len(_prompt_input) == 1 and isinstance(_prompt_input[0], str):
                actual_prompt = _prompt_input[0]
            else:
                print(f"Error: Invalid prompt format in edit_image: {_prompt_input}. Expected string or list/tuple of one string.")
                return (image, "Error: Invalid prompt format during execution.") # Return original image on error
        elif isinstance(_prompt_input, str):
            actual_prompt = _prompt_input
        else:
            print(f"Error: Invalid prompt type in edit_image: {type(_prompt_input)}. Expected string or list/tuple of one string.")
            return (image, "Error: Invalid prompt type during execution.") # Return original image on error
            
        try:
            set_seed(seed)
            inferencer = model["inferencer"]
            pil_image = tensor_to_pil(image)
            pil_image = pil_img2rgb(pil_image)
            inference_hyper = {
                "max_think_token_n": 1024 if show_thinking else 1024,
                "do_sample": False if not show_thinking else False,
                "text_temperature": text_temperature if show_thinking else 0.3,
                "cfg_text_scale": cfg_text_scale,
                "cfg_img_scale": cfg_img_scale,
                "cfg_interval": [cfg_interval, 1.0],
                "timestep_shift": timestep_shift,
                "num_timesteps": num_timesteps,
                "cfg_renorm_min": cfg_renorm_min,
                "cfg_renorm_type": cfg_renorm_type,
            }

            # DEBUG: Print information before the inferencer call
            print(f"DEBUG BagelImageEdit: pil_image type: {type(pil_image)}")
            print(f"DEBUG BagelImageEdit: inference_hyper keys: {list(inference_hyper.keys())}")
            if 'image' in inference_hyper:
                print("ERROR DEBUG BagelImageEdit: 'image' key FOUND in inference_hyper!")

            result = inferencer(image=pil_image, text=actual_prompt, think=show_thinking, **inference_hyper)
            
            # Convert image format - potentially a list of images
            output_image_or_images = result["image"]

            if isinstance(output_image_or_images, list):
                if not output_image_or_images:
                    print("Warning: Inferencer returned an empty list of edited images.")
                    return (image, "Error: No edited images generated.") # Return original on error

                tensor_images_list = [pil_to_tensor(img_item) for img_item in output_image_or_images]
                if not tensor_images_list:
                     return (image, "Error: Failed to convert edited images to tensors.")

                final_tensor_image = torch.cat(tensor_images_list, dim=0)
                print(f"Generated edited image sequence with {len(output_image_or_images)} frames. Final shape: {final_tensor_image.shape}")
            elif isinstance(output_image_or_images, Image.Image):
                final_tensor_image = pil_to_tensor(output_image_or_images)
                print(f"Edited single image with size: {output_image_or_images.size}. Final shape: {final_tensor_image.shape}")
            else:
                print(f"Error: Inferencer returned an unexpected edited image type: {type(output_image_or_images)}")
                return (image, "Error: Unexpected edited image data from model.") # Return original

            # Get reasoning process
            thinking_text = result.get("text", "") if show_thinking else ""

            return (final_tensor_image, thinking_text)

        except Exception as e:
            print(f"Error in image editing: {e}")
            # Return original image and error message
            return (image, f"Error: {str(e)}")


class BagelImageUnderstanding:
    """BAGEL Image Understanding Node"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("BAGEL_MODEL", {"tooltip": "BAGEL model"}),
                "image": ("IMAGE", {"tooltip": "Input image"}),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "What do you see in this image?",
                        "tooltip": "Question text",
                    },
                ),
            },
            "optional": {
                "show_thinking": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Display reasoning process"},
                ),
                "do_sample": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Enable sampling"},
                ),
                "text_temperature": (
                    "FLOAT",
                    {
                        "default": 0.3,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Text generation temperature",
                    },
                ),
                "max_new_tokens": (
                    "INT",
                    {
                        "default": 512,
                        "min": 64,
                        "max": 4096,
                        "step": 64,
                        "tooltip": "Maximum new tokens",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "understand_image"
    CATEGORY = "BAGEL/Core"

    def understand_image(
        self,
        model: Dict[str, Any],
        image: torch.Tensor,
        prompt: str,
        show_thinking: bool = False,
        do_sample: bool = False,
        text_temperature: float = 0.3,
        max_new_tokens: int = 512,
    ) -> Tuple[str]:
        _prompt_input = prompt
        actual_prompt: str

        if _prompt_input is None: actual_prompt = ""
        elif isinstance(_prompt_input, (list, tuple)):
            if len(_prompt_input) == 1 and isinstance(_prompt_input[0], str): actual_prompt = _prompt_input[0]
            else:
                print(f"Error: Invalid prompt format in understand_image: {_prompt_input}.")
                return (f"Error: Invalid prompt format during execution.",)
        elif isinstance(_prompt_input, str): actual_prompt = _prompt_input
        else:
            print(f"Error: Invalid prompt type in understand_image: {type(_prompt_input)}.")
            return (f"Error: Invalid prompt type during execution.",)

        try:
            # Removed set_seed(seed) call as 'seed' is not a parameter here
            inferencer = model["inferencer"]
            pil_image = tensor_to_pil(image)
            pil_image = pil_img2rgb(pil_image)
            inference_hyper = {
                "do_sample": do_sample,
                "text_temperature": text_temperature,
                "max_think_token_n": max_new_tokens,
            }

            # DEBUG: Print information before the inferencer call
            print(f"DEBUG BagelImageUnderstanding: pil_image type: {type(pil_image)}")
            print(f"DEBUG BagelImageUnderstanding: inference_hyper keys: {list(inference_hyper.keys())}")
            if 'image' in inference_hyper:
                print("ERROR DEBUG BagelImageUnderstanding: 'image' key FOUND in inference_hyper!")
            
            result = inferencer(image=pil_image, text=actual_prompt, think=show_thinking, understanding_output=True, **inference_hyper)
            
            answer_text = result["text"]
            print(f"Image understanding completed, response length: {len(answer_text)}")
            return (answer_text,)

        except Exception as e:
            print(f"Error in image understanding: {e}")
            return (f"Error: {str(e)}",)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "BagelModelLoader": BagelModelLoader,
    "BagelTextToImage": BagelTextToImage,
    "BagelImageEdit": BagelImageEdit,
    "BagelImageUnderstanding": BagelImageUnderstanding,
}

# Display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "BagelModelLoader": "BAGEL Model Loader",
    "BagelTextToImage": "BAGEL Text to Image",
    "BagelImageEdit": "BAGEL Image Edit",
    "BagelImageUnderstanding": "BAGEL Image Understanding",
}

# Export for ComfyUI
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# --- Progress bar integration for inference methods --- 
# We need a way to pass the progress bar to the inferencer if it supports it.
# The commit https://github.com/neverbiasu/ComfyUI-BAGEL/pull/23/commits/d1f4f2390cbd4fb3f61ef0ffa7bdf50dcf0c45a7
# modifies the inferencer and bagel model itself to accept a progress_callback.
# Assuming the inferencer has been updated similarly to accept a progress_callback:

# def _call_inferencer_with_progress(inferencer, pbar_total_steps, **kwargs):
#     pbar = comfy.utils.ProgressBar(pbar_total_steps)
#     current_step = 0
#     def progress_callback_wrapper(step, total_steps_from_inferencer, description=""):
#         nonlocal current_step
#         node_progress_step = int((step / total_steps_from_inferencer) * pbar_total_steps)
#         if node_progress_step > current_step:
#              current_step = node_progress_step
#         pbar.update_absolute(current_step, pbar_total_steps, description if description else None)
# 
#     kwargs["progress_callback"] = progress_callback_wrapper
#     result = inferencer(**kwargs)
#     pbar.update_absolute(pbar_total_steps, pbar_total_steps, "Done!")
#     return result
