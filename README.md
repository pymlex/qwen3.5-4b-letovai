# Qwen3.5-4B LetovAI

## Overview

This model is a LoRA adapter for `tvall43/Qwen3.5-4B-heretic`, fine-tuned to generate text in the distinct style of Egor Letov. We use a heretic version to bypass alignment restrictions when handling sensitive topics, similar to the style Letov used in his lyrics.

## Training data

The `pymlex/gr-oborona-lyrics` dataset contains Russian lyric texts collected for this project and filtered for Letov-related material.

## Training setup

- Framework: Unsloth + TRL
- Fine-tuning method: LoRA
- Sequence length: 1024
- LoRA rank: 16
- LoRA alpha: 32
- LoRA dropout: 0.05
- Optimizer: `adamw_8bit`
- Batch size: 8
- Gradient accumulation: 8
- Epochs: 3
- Learning rate: 3 epochs with `2e-4` and 2 with `5e-5`

## Inference

Get the model from Hugging Face.

```python
from peft import PeftModel
from unsloth import FastLanguageModel
import torch

BASE_MODEL_ID = "tvall43/Qwen3.5-4B-heretic"
ADAPTER_ID = "pymlex/qwen3.5-4b-letovai"

max_seq_length = 2048

base_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL_ID,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    dtype=None,
)

model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
model = FastLanguageModel.for_inference(model)
```

Generate poems and songs.

```python
SYSTEM_PROMPT = "Ты пишешь только песни и стихи в духе Егора Летова. Отвечай только художественным текстом."

def generate_letov_song(title, max_new_tokens=1536, temperature=0.9, top_p=0.95):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Напиши песню или стихотворение.\nТема: {title}\nПиши только текст.",
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

print(generate_letov_song("Будущее матушки Руси"))
```


## Evaluation

We evaluated the model with a pairwise **LLM-as-a-judge** benchmark, where a language model acts as the judge and compares two candidate texts, deciding which one is real and which one is synthetic. We generated poems and songs on 100 prompts built from 20 topics, 5 moods each, then compared the baseline `tvall43/Qwen3.5-4B-heretic` against the LoRA-tuned `pymlex/qwen3.5-4b-letovai` using `openai/gpt-oss-120b` as the judge. Each pair was judged 3 times, and the final result was tested with a two-sided binomial test. 

The fine-tuned model was chosen as real in 168/300 cases (56.0%), versus 132/300 for the baseline (44.0%). The difference was statistically significant at `p = 0.0431`. In fact, we ran this test several times for different models and the rate of "ft chosen as real" was between 0.53 and 0.6, so we definitely biased the result to the positive outcome, but it's not marvelous yet.
