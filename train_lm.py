from transformers import (
    set_seed,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
)
from huggingface_hub import login, create_repo
from datasets import Dataset, DatasetDict
from IPython.display import display, HTML
import json

TRAIN_DATA_FILE = "/kaggle/input/ga-clip_data/lm_train.jsonl"
OUTPUT_DIR = "/kaggle/working"
MODEL_NAME = "t5-small"
API_KEY = ""
HUB_MODEL_ID = "mghiasvand1/GA-CLIP_lm"
SEED = 1
MAX_SOURCE_LENGTH = 85
MAX_TARGET_LENGTH = 100
TRAINING_ARGS = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=16,
    num_train_epochs=5,
    learning_rate=3e-4,
    logging_strategy="epoch",
    save_strategy="epoch",
    report_to=[],
    seed=SEED,
)


def load_jsonl(filepath):
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line.strip())
            data.append({"source": record["source"], "target": record["target"]})
    return data


display(
    HTML(
        "<script>Jupyter.notebook.kernel.execute('config NotebookApp.iopub_msg_rate_limit=10000000000')</script>"
    )
)
login(token=API_KEY)
set_seed(SEED)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
train_data = load_jsonl(TRAIN_DATA_FILE)
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
    create_repo(HUB_MODEL_ID, private=True, exist_ok=True)
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    trained_model = AutoModelForSeq2SeqLM.from_pretrained(OUTPUT_DIR)
    trained_model.push_to_hub(HUB_MODEL_ID)
    tokenizer.push_to_hub(HUB_MODEL_ID)