import argparse
from datasets import get_dataset
from models import resolve_model_name, MODEL_ALIASES
from evaluator_revised import Evaluator

def get_args():
    p = argparse.ArgumentParser(description="Qwen‑VL Evaluation")
    p.add_argument("--model_name", default="qwen2vl-2b",
                   help=("model alias or HF path. supporting alias: "
                         f"{', '.join(MODEL_ALIASES.keys())}"))
    p.add_argument("--dataset", choices=["MNIST", "CIFAR10", "CIFAR100"],
                   default="CIFAR10", help="test dataset")
    p.add_argument("--is_instruct", type=int, choices=[0, 1], default=0,
                   help="1 if the model is an instruct model, otherwise 0")
    p.add_argument("--gpus", default="0",
                   help="'0', '0,1' etc. '', 'cpu' for cpu inference")
    p.add_argument("--batch_size", type=int, default=8, help="batch size")
    p.add_argument("--output_dir", default="./results", help="output folder")
    p.add_argument("--wandb_project", default="vlm-forgetting", help="name of the wandb project")
    p.add_argument("--wandb_group_name", default=None, help="group name of the wandb project")
    p.add_argument("--wandb_run_name", default=None, help="run name of the wandb project")
    return p.parse_args()

def main():
    args = get_args()
    hf_id = resolve_model_name(args.model_name)

    dataset, loader, classes = get_dataset(args.dataset, batch=args.batch_size)
    evaluator = Evaluator(hf_id, args.is_instruct, args.gpus, args.wandb_project, args.wandb_group_name, args.wandb_run_name)
    evaluator.evaluate(dataset, loader, classes, args.output_dir)

if __name__ == "__main__":
    main()
