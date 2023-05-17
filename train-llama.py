from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
from transformers.models.llama.tokenization_llama import LlamaTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import load_from_disk
import random
import logging
import sys
import argparse
import os
import torch
import boto3
import subprocess
#import bitsandbytes as bnb

#download requirements
s3 = boto3.client('s3')
#s3.download_file('sagemaker-us-east-1-685382160364','python/env/requirements.txt','/opt/ml/code/requirements.txt')

#install
#subprocess.check_call([sys.executable,"-m","pip","install","-r","/opt/ml/code/requirements.txt"])

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    #parser.add_argument("--eval_steps",type=int,default=5000)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--learning_rate", type=str, default=2e-5)
    parser.add_argument("--evaluation_strategy",type=str,default="epoch")
    parser.add_argument("--gradient_accumulation_steps",type=int,default=1)
    parser.add_argument("--c",type=bool,default=False)
    #parser.add_argument("--logging_steps",type=int,default=5000)
    parser.add_argument("--save_steps",type=int,default=500)
    parser.add_argument("--save_strategy",type=str,default="steps")
    parser.add_argument("--save_total_limit",type=int,default=4)
    parser.add_argument("--model_max_length",type=int,default=512)
    parser.add_argument("--bf16",type=bool,default=True)

    # Data, model, and output directories
    parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])
    #parser.add_argument("--deepspeed_config", type=str, default=None, help="Path to deepspeed config file.")
    #parser.add_argument("--deepspeed", type=str, default=None, help="Path to deepspeed config file.")
    
#     # distributed training
#     parser.add_argument('--pipeline-model-parallel-size', type=int, default=16, help='Degree of pipeline model parallelism.')
#     parser.add_argument('--distributed-backend', default='nccl', choices=['nccl', 'gloo', 'ccl'],)
#     parser.add_argument('--tensor-model-parallel-size', type=int, default=1, help='Degree of tensor model parallelism.')
    
    args, _ = parser.parse_known_args()

    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    #training_dir = '/opt/ml/input/data/train'
    #test_dir = '/opt/ml/input/data/test'
    # load datasets
    train_dataset = load_from_disk(args.training_dir)
    test_dataset = load_from_disk(args.test_dir)

    logger.info(f" loaded train_dataset length is: {len(train_dataset)}")
    logger.info(f" loaded test_dataset length is: {len(test_dataset)}")

    # compute metrics function for binary classification
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

    # download model from model hub
    model = AutoModelForCausalLM.from_pretrained(args.model_name,use_cache=False)#os.environ.get('SM_CHANNEL_MODEL',None))
    #tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer = LlamaTokenizer.from_pretrained(args.model_name, model_max_length=args.model_max_length, padding_side="right", padding='max_length', truncation=True)
    #tokenizer = LlamaTokenizer.from_pretrained(args.model_name,truncation=True)
         
    tokenizer.add_special_tokens({'additional_special_tokens': ['[STOP]','[SEP]']})
    if tokenizer.pad_token is None:
        print("-----------no pad token and add special token PAD----")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    #tokenizer.pad_token = tokenizer.eos_token
    #tokenizer.add_special_tokens({'additional_special_tokens': ['[STOP]','[SEP]']})
    
    # define training args
    training_args = TrainingArguments(
        output_dir="/tmp/intermediate",
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        warmup_steps=args.warmup_steps,
        #evaluation_strategy=args.evaluation_strategy,#"epoch",
        logging_dir=f"{args.output_data_dir}/logs",
        max_steps = 200,
        #eval_steps  = 60,
        evaluation_strategy = "no",
        logging_steps = 100,
        gradient_checkpointing=True,
        learning_rate=float(args.learning_rate),
        #save_steps = args.save_steps,
        save_strategy = "no",
        save_total_limit = 2,
        save_on_each_node = True,
        #fsdp="full_shard",
        #save
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        fp16=True,  
    )

    
    #optimizer = bnb.optim.Adam8bit(model.parameters(), lr=args.learning_rate)
    
    # create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        #compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer
        #is_model_parallel=True
    )

    # train model
    trainer.train()

    # evaluate model
    eval_result = trainer.evaluate(eval_dataset=test_dataset)

    # writes eval result to file which can be accessed later in s3 ouput
    with open(os.path.join(args.output_data_dir, "eval_results.txt"), "w") as writer:
        print(f"***** Eval results *****")
        for key, value in sorted(eval_result.items()):
            writer.write(f"{key} = {value}\n")

    # Saves the model to s3
    tokenizer.save_pretrained(args.model_dir)
    trainer.save_model(args.model_dir)
