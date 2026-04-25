"""Dummy example: quantize ViT-B/16 and evaluate on ImageNet."""

import argparse

import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from quantize import (
    load_pretrained_vit,
    quantize_model,
    QuantizedLinear,
    InputQuantizedWrapper,
    find_quantized_layers,
    GPTQLinear,
    QuantizedConv2d,
    quantize_depthwise_conv2d,
)
from timm.layers import LayerNorm2d


BATCH_SIZE = 64
NUM_WORKERS = 4


def evaluate(model, dataloader, device):
    """Compute top-1 and top-5 accuracy."""
    model.eval()
    correct1, correct5, total = 0, 0, 0
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        for images, targets in pbar:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            _, pred5 = outputs.topk(5, dim=1)
            _, pred1 = outputs.max(1)
            total += targets.size(0)
            correct1 += (pred1 == targets).sum().item()
            correct5 += (pred5 == targets.unsqueeze(1)).any(dim=1).sum().item()
            pbar.set_postfix(top1=f"{100.0 * correct1 / total:.2f}%", top5=f"{100.0 * correct5 / total:.2f}%")
    return 100.0 * correct1 / total, 100.0 * correct5 / total


def main():
    parser = argparse.ArgumentParser(description="Quantize ViT-B/16 and evaluate on ImageNet")
    parser.add_argument(
        "imagenet_dir",
        type=str,
        help="Path to ImageNet validation directory (e.g. /path/to/imagenet/val)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="vit_base_patch16_224",
        help="timm model name (default: vit_base_patch16_224)",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=8,
        help="Number of bits for quantization (default: 8)",
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Skip quantization; evaluate baseline model in full precision",
    )
    parser.add_argument(
        "--quant-type",
        type=str,
        default="linear",
        help=(
            "Type of quantization to run\n"
            "* linear     – symmetric per-channel weight + per-token activation (nn.Linear only)\n"
            "* conv2d     – symmetric per-channel weight + per-token activation (nn.Conv2d only)\n" 
            "* depthwise  – symmetric per-channel weight + per-token activation (depthwise nn.Conv2d only)\n"
            "* absmax     – symmetric per-channel weight + per-token activation (nn.Linear and nn.Conv2d)\n" 
            "* asymm      – symmetric quantizaion for weights, asymmetric quantization for inputs (nn.Linear and nn.Conv2d)\n"
            "* all        – symmetric quantization of nn.Linear, nn.Conv2d, nn.LayerNorm\n"
            "* gptq       – GPTQ Hessian-guided quantization of nn.Linear layers\n"
            "* layernorm  – wrap LayerNorm with symmetric wrapper"
            "(default: linear)"
        ),
    )
    parser.add_argument(
        "--gptq-calib-batches",
        type=int,
        default=64,
        help="Number of calibration batches for GPTQ Hessian estimation (default: 64)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading pre-trained {args.model} from timm...")
    model = load_pretrained_vit(args.model, pretrained=True)
    model = model.to(device)


    print("Loading ImageNet validation set...")
    data_config = timm.data.resolve_data_config(model.pretrained_cfg)
    preprocess = timm.data.create_transform(**data_config)
    val_dataset = ImageFolder(args.imagenet_dir, transform=preprocess)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )
    print(f"Validation samples: {len(val_dataset)}")
    
    if args.no_quantize:
        print("Skipping quantization (baseline full precision)")
    # symmetric quant to linear
    elif args.quant_type == "linear":
        print(f"Quantizing nn.Linear layers to {args.bits}-bit...")
        quantize_model(model, [(nn.Linear, QuantizedLinear, {"bits": args.bits})])
        replaced = find_quantized_layers(model, QuantizedLinear)
        print(f"Quantized {len(replaced)} layers to {args.bits}-bit")
    # symmetric quant to conv2d
    elif args.quant_type == "conv2d":
        print(f"Quantizing nn.Conv2d layers to {args.bits}-bit...")
        quantize_model(model, [(nn.Conv2d, QuantizedConv2d, {"bits": args.bits})])
        replaced = find_quantized_layers(model, QuantizedConv2d)
        print(f"Quantized {len(replaced)} layers to {args.bits}-bit")
    # symmetric quantization to depthwise conv2d only
    elif args.quant_type == "depthwise":
        print(f"Quantizing depthwise nn.Conv2d layers to {args.bits}-bit...")
        quantize_depthwise_conv2d(
            model,
            bits=args.bits,
            asymmetric_acts=False,
        )
        replaced = find_quantized_layers(model, QuantizedConv2d)
        print(f"Quantized {len(replaced)} depthwise layers to {args.bits}-bit")
        if len(replaced) > 0:
            print("First few quantized depthwise layers:")
            for i, name in enumerate(replaced.keys()):
                if i >= 10:
                    break
                print(f"  {name}")
    # symmetric quant to linear & conv2d
    elif args.quant_type == "absmax":
        print(f"Quantizing nn.Linear, nn.Conv2d layers to {args.bits}-bit...")
        quantize_model(model, [(nn.Linear, QuantizedLinear, {"bits": args.bits}),
                               (nn.Conv2d, QuantizedConv2d, {"bits": args.bits}),])
        replaced = find_quantized_layers(model, QuantizedLinear)
        replaced.update(find_quantized_layers(model, QuantizedConv2d))
        print(f"Quantized {len(replaced)} layers to {args.bits}-bit")
    # symmetric quantization for weights, asymmetric quantizaion for inputs
    elif args.quant_type == "asymm":
        print(f"Quantizing nn.Linear, nn.Conv2d layers to {args.bits}-bit...")
        quantize_model(model, [(nn.Linear, QuantizedLinear, {"bits": args.bits, "asymmetric_acts": True}),
                               (nn.Conv2d, QuantizedConv2d, {"bits": args.bits, "asymmetric_acts": True}), ])
        replaced = find_quantized_layers(model, QuantizedLinear)
        replaced.update(find_quantized_layers(model, QuantizedConv2d))
        print(f"Quantized {len(replaced)} layers to {args.bits}-bit")
    # symmetric quantization of nn.Linear, nn.Conv2d, nn.LayerNorm todo ?
    elif args.quant_type == "all":
        print(f"Quantizing nn.Linear, nn.Conv2d, nn.LayerNorm layers to {args.bits}-bit...")
        quantize_model(model, [(nn.Linear, QuantizedLinear, {"bits": args.bits}), 
                                (nn.Conv2d, InputQuantizedWrapper, {"bits": args.bits}),
                                 (nn.LayerNorm, InputQuantizedWrapper, {"bits": args.bits})])
        replaced = find_quantized_layers(model, QuantizedLinear)
        replaced.update(find_quantized_layers(model, InputQuantizedWrapper))
        print(f"Quantized {len(replaced)} layers to {args.bits}-bit")
    # GPTQ Hessian-guided quantization of nn.Linear layers
    elif args.quant_type == "gptq":
        print(f"Quantizing nn.Linear layers to {args.bits}-bit using GPTQ...")
        print(f"  Calibration batches : {args.gptq_calib_batches}  "
              f"(= {args.gptq_calib_batches * BATCH_SIZE} samples)")
 
        # Step 1: replace all nn.Linear with GPTQLinear and start hooks
        quantize_model(model, [(nn.Linear, GPTQLinear, {"bits": args.bits})])
        gptq_layers = find_quantized_layers(model, GPTQLinear)
        for layer in gptq_layers.values():
            layer.start_calibration()
 
        # Step 2: run calibration forward passes (no grad needed)
        model.eval()
        print("  Running calibration passes...")
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(
                tqdm(val_loader, total=args.gptq_calib_batches, desc="Calibrating")
            ):
                if batch_idx >= args.gptq_calib_batches:
                    break
                model(images.to(device))
 
        # Step 3: finalize – run the per-layer GPTQ weight update
        print("  Running GPTQ weight updates...")
        for name, layer in tqdm(gptq_layers.items(), desc="GPTQ optimizing"):
            layer.finish_calibration()
 
        print(f"GPTQ quantized {len(gptq_layers)} layers to {args.bits}-bit")
    # wrap LayerNorm with symmetric wrapper
    elif args.quant_type == "layernorm":
        print(f"Wrapping nn.LayerNorm2d layers to {args.bits}-bit...")
        quantize_model(model, [], [(LayerNorm2d, {"bits": args.bits})])
        replaced = find_quantized_layers(model, InputQuantizedWrapper)
        print(f"Quantized {len(replaced)} layers to {args.bits}-bit")
    else:
        raise ValueError(
            f"Unknown --quant-type '{args.quant_type}'. "
            "Choose from: linear / conv2d / depthwise / absmax / asymm / all / gptq / layernorm"
        )
    print(model)

    
    print("Running evaluation...")
    top1, top5 = evaluate(model, val_loader, device)
    print(f"Top-1 accuracy: {top1:.2f}%")
    print(f"Top-5 accuracy: {top5:.2f}%")


if __name__ == "__main__":
    main()
