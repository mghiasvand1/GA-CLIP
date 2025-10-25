from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
    set_seed,
)
from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import login, create_repo

API_KEY = ""
SEED = 1
MAX_SOURCE_LENGTH = 85
MAX_TARGET_LENGTH = 100
TRAINING_ARGS = TrainingArguments(
    output_dir="/kaggle/working",
    per_device_train_batch_size=16,
    num_train_epochs=5,
    learning_rate=3e-4,
    logging_strategy="epoch",
    save_strategy="epoch",
    report_to=[],
    seed=SEED,
)
login(token=API_KEY)
set_seed(SEED)
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("t5-small")
dataset = load_dataset(
    "mghiasvand1/GA-CLIP_data", data_files="lm_train.jsonl", split="train"
)
train_data = [
    {"source": record["source"], "target": record["target"]} for record in dataset
]
train_dataset = Dataset.from_list(train_data)
ds = DatasetDict({"train": train_dataset})


def preprocess(batch):
    model_inputs = tokenizer(
        batch["source"],
        max_length=MAX_SOURCE_LENGTH,
        padding="max_length",
        truncation=True,
    )
    labels = tokenizer(
        text_target=batch["target"],
        max_length=MAX_TARGET_LENGTH,
        padding="max_length",
        truncation=True,
    )
    model_inputs["labels"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels["input_ids"]
    ]
    return model_inputs


def fine_tune():
    tokenized = ds.map(
        preprocess, batched=True, remove_columns=ds["train"].column_names
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = Trainer(
        model=model,
        args=TRAINING_ARGS,
        train_dataset=tokenized["train"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    create_repo("mghiasvand1/GA-CLIP_lm", private=True, exist_ok=True)
    trainer.train()
    trainer.save_model("/kaggle/working")
    tokenizer.save_pretrained("/kaggle/working")
    trained_model = AutoModelForSeq2SeqLM.from_pretrained("/kaggle/working")
    trained_model.push_to_hub("mghiasvand1/GA-CLIP_lm")
    tokenizer.push_to_hub("mghiasvand1/GA-CLIP_lm")
