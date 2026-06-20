# Thomas Cole Style Transfer — FLUX.1-schnell LoRA

Fine-tune FLUX.1-schnell to paint like Thomas Cole. This repository contains the training configuration and inference notebook for `thcole_flux_v1`, a LoRA adapter that transfers the Hudson River School aesthetic — dramatic skies, luminous wilderness, and romanticist oil-painting textures — to any subject.

**Trigger word:** `THCOLE style`

---

## What's inside

| Section | Description |
|---|---|
| **1. Training Configuration** | Documented `ai-toolkit` job spec used to produce `thcole_flux_v1.safetensors` — not re-executed, included for reproducibility |
| **2. Inference Setup** | Load FLUX.1-schnell and mount the trained LoRA from Google Drive |
| **3. Generation** | Text-to-image and image-to-image pipelines with live latency measurement |
| **4. Evaluation** | CLIP ViT-L/14 similarity scoring — computed live, not asserted |

---

## Model details

| Parameter | Value |
|---|---|
| Base model | `black-forest-labs/FLUX.1-schnell` (12B, CFG-distilled) |
| LoRA rank | 16 (alpha 16, effective scale 1.0) |
| Training steps | 1,000 |
| Optimizer | AdamW 8-bit (`bitsandbytes`), lr = 1e-4 |
| Resolution | Multi-res bucketing — 512, 768, 1024 |
| Dataset | 20+ curated Thomas Cole paintings |
| Hardware | NVIDIA A100 80GB (Google Colab) |
| Training time | ~2 hours |

### Design decisions

**LoRA over full fine-tuning** — FLUX.1-schnell's 12B parameters require >160GB VRAM for full fine-tuning and risk catastrophic forgetting. LoRA trains ~80MB of low-rank adapters injected into the attention layers while keeping the base model's general knowledge intact.

**Trigger word `THCOLE style`** — deliberately distinct from "Thomas Cole" to avoid colliding with the base model's existing pretrained association with the artist's name, giving the LoRA a clean, dedicated activation hook.

**Multi-resolution bucketing** — Cole's paintings are predominantly wide panoramic landscapes. Training across [512, 768, 1024] prevents overfitting to a single fixed aspect ratio.

**AdamW 8-bit + gradient checkpointing** — cuts optimizer state memory ~75% vs standard AdamW. Gradient checkpointing trades recomputation for VRAM, allowing larger batch/resolution combinations to fit on the A100.

---

## Inference

FLUX.1-schnell is CFG-distilled. Use `guidance_scale=1.0` and `num_inference_steps=4` — these are not arbitrary, they are the values the distillation was optimized for. Higher guidance or more steps degrade output quality on this model.

### Text-to-image

```python
from diffusers import FluxPipeline
import torch

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()
pipe.load_lora_weights("path/to/thcole_flux_v1.safetensors")

image = pipe(
    "A vast American wilderness valley in the THCOLE style, "
    "Catskill mountains, winding river, autumn foliage, golden hour",
    guidance_scale=1.0,
    num_inference_steps=4,
    max_sequence_length=256,
    generator=torch.Generator("cpu").manual_seed(42),
).images[0]
```

### Image-to-image (style transfer)

```python
from diffusers import FluxImg2ImgPipeline
from diffusers.utils import load_image

pipe_i2i = FluxImg2ImgPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16,
)
pipe_i2i.enable_model_cpu_offload()
pipe_i2i.load_lora_weights("path/to/thcole_flux_v1.safetensors")

init_image = load_image("your_photo.jpg").convert("RGB").resize((1024, 768))

result = pipe_i2i(
    prompt="A majestic landscape in the THCOLE style, dramatic oil painting, "
           "Hudson River School aesthetic",
    image=init_image,
    strength=0.65,       # 0.6–0.7 is the sweet spot for recognizable-but-stylized output
    guidance_scale=1.0,
    num_inference_steps=4,
    generator=torch.Generator("cpu").manual_seed(42),
).images[0]
```

`strength` controls how much of the source photo's structure survives — 0.1 stays close to the original, 0.9 is near-total repaint.

---

## Evaluation

Style adherence is measured using CLIP ViT-L/14 (the same encoder family used in FLUX's own text conditioning). The notebook computes two metrics:

**Prompt similarity** — image/text cosine similarity for each generated image against its prompt. Quantifies how well each generation matches its intended description.

**Style adherence** — CLIP similarity against Hudson River School style descriptors vs. a "a photograph" control. Isolates whether the LoRA injects style rather than generic scene composition. The control descriptor should score lowest when style transfer is working.

Scores are produced live by running the notebook and will vary slightly between runs depending on GPU allocation, prompts, and seeds used.

---

## Requirements

```
diffusers
transformers
accelerate
peft
sentencepiece
protobuf
safetensors
torchao
clip @ git+https://github.com/openai/CLIP.git
```

Install:

```bash
pip install diffusers transformers accelerate peft sentencepiece protobuf safetensors torchao
pip install git+https://github.com/openai/CLIP.git
```

**Note on numpy:** Do not force-install `numpy<2`. Current Colab images ship numpy 2.x, which is compatible with current diffusers/FLUX. Forcing a downgrade causes binary-incompatibility crashes against other preinstalled packages (opencv, jax, cupy) compiled against numpy 2.x. If you previously ran a numpy downgrade in the session, restart the runtime before running inference cells.

---

## Re-training from scratch

The notebook documents but does not re-execute training — `thcole_flux_v1.safetensors` was produced in a separate 1,000-step session (~2 hrs on A100). To retrain:

1. Prepare a dataset of 20+ captioned images in `/content/dataset`, each with a `.txt` caption containing `THCOLE style`.
2. Install [ai-toolkit](https://github.com/ostris/ai-toolkit).
3. Uncomment the `run_job(training_job_config)` call in Section 1 and execute.

---

## About Thomas Cole

Thomas Cole (1801–1848) was the founder of the Hudson River School, America's first major landscape painting movement. His work is known for its sweeping panoramic scale, luminous atmospheric light, dramatic cloud formations, and the tension between civilization and wilderness — themes that recur across his *Course of Empire* series, *The Oxbow*, and his Catskill Mountain landscapes. This LoRA was trained to capture those visual signatures: rich impasto texture, golden-hour color palettes, theatrical skies, and a sense of sublime scale.

---

## License

The LoRA weights and notebook are released under MIT. FLUX.1-schnell is subject to the [Black Forest Labs FLUX.1 License](https://huggingface.co/black-forest-labs/FLUX.1-schnell/blob/main/LICENSE.md). Thomas Cole's paintings are in the public domain.
