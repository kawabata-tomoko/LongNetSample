import numpy as np
import torch
from longnetLM.config.config import *
from longnetLM.config.LongNetTokenizer import LongNetTokenizer
from sklearn.metrics import precision_recall_fscore_support
from transformers import DefaultDataCollator, Trainer, TrainingArguments
from longnetLM.srcs.LongNetForSeqCls import LongNetEncoderForSequenceClassification,LongNetDecoderForSequenceClassification


def compute_metrics(p):
    logits,labels= p
    pred=np.argmax(logits, axis=-1)
    precision, recall, fscore, support = precision_recall_fscore_support(labels, pred, average="weighted")
    return {"precision":precision,"recall":recall,"fscore":fscore}

def p_count(m):
    ttp=0
    tp=0
    for p in m.parameters():
        c=p.numel()
        if p.requires_grad == True:
            ttp+=c
        tp+=c
    print(f"Total trainable parameters: {ttp}")
    print(f"Total parameters: {tp}")

vocab_path = "./longnetLM/config/vocab.txt"
tokenizer = LongNetTokenizer(vocab_path)
config = LongnetConfig(
    attention_type="MultiHeadAttention",
    # attention_type="DilatedAttention",
    vocab_size=tokenizer.vocab_size,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    mask_token_id=tokenizer.mask_token_id,
    flash_attention=True,
    segment_length='[2048,4096,8192,16384]', 
    dilated_ratio='[1,2,4,8]', 
    num_labels=13
    )

# model = LongNetEncoderForSequenceClassification(config=config,torch_dtype=torch.bfloat16)
model = LongNetDecoderForSequenceClassification(config=config,torch_dtype=torch.bfloat16)

p_count(model)
datacollator = DefaultDataCollator()
     
training_args=TrainingArguments(
    output_dir=f"{YOUR_OUTPUT_PATH_HERE}",
    evaluation_strategy="steps",
    gradient_checkpointing=True,
    eval_steps=10,
    save_steps=10,
    save_total_limit=10,
    learning_rate=5e-4,# 0.000005,#EVO use 0.00009698,
    lr_scheduler_type= "cosine",
    warmup_ratio = 0.1,
    weight_decay=0.1,#EVO use 0.1
    num_train_epochs=50,#EVO use 10
    gradient_accumulation_steps=4,#pretrained 8
    per_device_train_batch_size=4,#pretrained 4
    per_device_eval_batch_size=4,#pretrained 4
    neftune_noise_alpha=10.0,
    max_grad_norm=5,
    bf16=True,
    logging_steps =1,
    report_to="wandb",
    optim="adamw_apex_fused",
)

dataset_path=f"{YOUR_DATASET_PATH_HERE}"

from datasets import load_from_disk
trainset=load_from_disk(f"{dataset_path}/trainset")
evalset=load_from_disk( f"{dataset_path}/evalset")
testset=load_from_disk( f"{dataset_path}/testset")



trainer=Trainer(
    model=model,
    args=training_args,
    train_dataset= trainset,
    eval_dataset= evalset,
    data_collator=datacollator,
    compute_metrics=compute_metrics
)


# trainer = accelerator.prepare(trainer)

trainer.train()
ans=trainer.predict(testset)
print(ans)
torch.save(ans,"./sample_ans.pth")