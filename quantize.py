"""
Quantization utilities for transformer models.
Loads pre-trained ViT from timm and provides a generic recursive
quantization function that replaces target layers with user-supplied classes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm


def load_pretrained_vit(model_name: str = "vit_base_patch16_224", pretrained: bool = True):
    """Load a pre-trained Vision Transformer from timm (ViT-B/16 by default)."""
    return timm.create_model(model_name, pretrained=pretrained)


class QuantizedLinear(nn.Module):
    """
    Standalone replacement for nn.Linear with per-output-channel symmetric
    fake quantization of the weight, and per-token symmetric fake quantization
    of activations with online scale.
    """

    def __init__(self, original: nn.Linear, bits: int = 8, asymmetric_acts: bool = False, **kwargs):
        """
        :param bits: number of bits to quantize to
        :param asymmetric_acts: False - symmetric (absmax) qaunt for both weights and inputs
                                True - symmetric (absmax) quant for the weights, asymmetric (zeropoint) quant for the inputs.
        """
        super().__init__()
        assert isinstance(original, nn.Linear), "QuantizedLinear expects nn.Linear"
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.bits = bits
        self.asymmetric_acts = asymmetric_acts
        self.weight = nn.Parameter(original.weight.data.clone())
        self.bias = nn.Parameter(original.bias.data.clone()) if original.bias is not None else None
        maxq = 2 ** (bits - 1) - 1
        self.register_buffer("maxq", torch.tensor(maxq, dtype=torch.float32))
        self._compute_scales()

    def _compute_scales(self):
        """Compute per-channel symmetric quantization scales from the weight."""
        w = self.weight.data
        # Per-output-channel: shape (out_features,)
        w_flat = w.reshape(w.shape[0], -1)
        w_max = w_flat.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        scale = w_max / self.maxq
        self.register_buffer("scale", scale.reshape(-1, *([1] * (w.dim() - 1))))

    def _fake_quant_weight(self):
        """Quantize and dequantize the weight (fake quantization)."""
        w = self.weight
        dev = w.device
        scale = self.scale.to(dev)
        maxq = self.maxq.to(dev)
        q = torch.clamp(torch.round(w / scale), -(maxq + 1), maxq)
        return (scale * q).to(w.dtype)

    def forward(self, x):
        # check quantization type for inputs (symmetric \ asymmetric)
        if self.asymmetric_acts:
            # quant range parameters
            maxq = self.maxq.to(x.device)
            qmin = -(maxq + 1)
            q_range = (2 * maxq) + 1
            x_min = x.amin(dim=-1, keepdim=True)
            x_max = x.amax(dim=-1, keepdim=True)
            x_max = torch.max(x_max, x_min + 1e-8)  # for the case x_min = x_max
            # calculate scale parameter and zeropoint
            scale_act = (x_max - x_min) / q_range
            z_act = torch.round(-x_min / scale_act) + qmin
            # the quantization
            q_act = torch.clamp(torch.round(x / scale_act + z_act), qmin, maxq)
            x_q = ((q_act - z_act) * scale_act).to(x.dtype)
        else:
            # 1. Online per-token activation scale
            x_flat = x.reshape(-1, x.shape[-1])  # (B*N, D) or (B, D)
            x_max = x_flat.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
            scale_act = x_max / self.maxq.to(x.device)
            scale_act = scale_act.reshape(*x.shape[:-1], 1)  # (B, N, 1) or (B, 1)
            # 2. Fake-quantize activation
            maxq = self.maxq.to(x.device)
            q_act = torch.clamp(torch.round(x / scale_act), -(maxq + 1), maxq)
            x_q = (scale_act * q_act).to(x.dtype)

        # 3. Fake-quantize weight and compute output
        w_q = self._fake_quant_weight()
        return F.linear(x_q, w_q, self.bias)


class InputQuantizedWrapper(nn.Module):
    """
    Wrapper that quantizes the input before passing it to the wrapped module.
    Uses per-token symmetric fake quantization with online scale.
    """

    def __init__(self, module: nn.Module, bits: int = 8, **kwargs):
        super().__init__()
        self.module = module
        self.bits = bits
        maxq = 2 ** (bits - 1) - 1
        self.register_buffer("maxq", torch.tensor(maxq, dtype=torch.float32))

    def _quantize_input(self, x):
        """Per-token symmetric fake quantization of the input."""
        x_flat = x.reshape(-1, x.shape[-1])
        x_max = x_flat.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale_act = x_max / self.maxq.to(x.device)
        scale_act = scale_act.reshape(*x.shape[:-1], 1)
        maxq = self.maxq.to(x.device)
        q_act = torch.clamp(torch.round(x / scale_act), -(maxq + 1), maxq)
        return (scale_act * q_act).to(x.dtype)

    def forward(self, x):
        x_q = self._quantize_input(x)
        return self.module(x_q)


class GPTQLinear(nn.Module):
    """
    nn.Linear replacement whose weights are optimally quantized using the
    GPTQ (Optimal Brain Quantization) algorithm.
 
    Usage
    -----
    1.  Create the layer:
            layer = GPTQLinear(original_linear, bits=4)
    2.  Feed calibration batches through the *original* model while the
        GPTQ hook is active:
            layer.start_calibration()
            for x, _ in calib_loader:
                model(x)           # forward pass accumulates H
            layer.finish_calibration()
    3.  Use normally at inference – weights are now GPTQ-quantized.
 
    Alternatively, call GPTQLinear.gptq_quantize_model() which wraps the
    whole replace + calibrate + finalize pipeline for you.
    """
 
    def __init__(self, original: nn.Linear, bits: int = 8, damp_pct: float = 0.01, **kwargs):
        super().__init__()
        assert isinstance(original, nn.Linear), "GPTQLinear expects nn.Linear"
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.bits = bits
        self.damp_pct = damp_pct
 
        # Quantization grid: symmetric, per-output-channel (same as Method 1)
        maxq = 2 ** (bits - 1) - 1
        self.register_buffer("maxq", torch.tensor(float(maxq)))
 
        # Copy weight & bias; weight will be overwritten after calibration
        self.weight = nn.Parameter(original.weight.data.clone(), requires_grad=False)
        if original.bias is not None:
            self.bias = nn.Parameter(original.bias.data.clone(), requires_grad=False)
        else:
            self.bias = None
 
        # Running Hessian accumulator  H = 2 Σ xᵢ xᵢᵀ  (in_features × in_features)
        self.register_buffer(
            "_H", torch.zeros(self.in_features, self.in_features)
        )
        self._n_samples: int = 0
        self._hook = None          # forward pre-hook handle
        self._calibrating: bool = False
 
    # ------------------------------------------------------------------
    # Calibration helpers
    # ------------------------------------------------------------------
 
    def start_calibration(self):
        """Register the forward pre-hook that accumulates the Hessian."""
        self._calibrating = True
        self._H.zero_()
        self._n_samples = 0
        self._hook = self.register_forward_pre_hook(self._accumulate_hessian)
 
    def _accumulate_hessian(self, _module, args):
        """Pre-hook: update H with the current input batch."""
        x = args[0].detach()                             # (B, *, d_in)
        x_2d = x.reshape(-1, self.in_features).float()  # (N, d_in)
        # Keep _H on the same device as the input (handles CPU → CUDA migration)
        if self._H.device != x_2d.device:
            self._H = self._H.to(x_2d.device)
        self._H += 2.0 * x_2d.t() @ x_2d
        self._n_samples += x_2d.shape[0]
 
    def finish_calibration(self):
        """Remove the hook and run the GPTQ weight update."""
        if self._hook is not None:
            self._hook.remove()
            self._hook = None
        self._calibrating = False
        if self._n_samples == 0:
            raise RuntimeError("finish_calibration called with zero calibration samples.")
        self._run_gptq()
        # Free the Hessian buffer — no longer needed
        self._H.zero_()
 
    # ------------------------------------------------------------------
    # Core GPTQ algorithm
    # ------------------------------------------------------------------
 
    def _run_gptq(self):
        """
        Quantize self.weight in-place using the accumulated Hessian.
 
        Column-wise (i.e. input-feature-wise) greedy OBQ update:
          for each column i (sorted by diagonal of H⁻¹ for stability):
              q[:,i] = quant(w[:,i])
              w[:,i+1:] -= outer(w[:,i] - q[:,i], H_inv[i, i+1:] / H_inv[i,i])
        We use the Cholesky decomposition of H⁻¹ for numerical stability
        (as in the original GPTQ paper).
        """
        dev = self.weight.device
        W = self.weight.data.float()          # (out, in)
        H = self._H.to(dev).float()
        self.maxq = self.maxq.to(dev)
 
        # 1. Dampen the Hessian to handle near-singular cases
        damp = self.damp_pct * H.diag().mean()
        H.diagonal().add_(damp)
 
        # 2. Invert H via Cholesky  (H = LLᵀ  =>  H⁻¹ available column-wise)
        try:
            L = torch.linalg.cholesky(H)                   # lower triangular
            H_inv = torch.cholesky_inverse(L)              # (in, in)
        except torch.linalg.LinAlgError:
            # Fallback: pseudo-inverse if Cholesky fails (degenerate Hessian)
            H_inv = torch.linalg.pinv(H)
 
        # 3. Column-wise quantization with error propagation
        #    We process columns left-to-right (column = input dimension).
        d_in = self.in_features
        W_q = W.clone()
 
        for i in range(d_in):
            w_col = W_q[:, i]                              # (out,)
 
            # Per-output-channel scale for this column
            w_max = w_col.abs().amax().clamp(min=1e-8)
            scale = w_max / self.maxq                      # scalar
 
            # Quantize column i
            q_col = (w_col / scale).round().clamp(-self.maxq - 1, self.maxq) * scale
 
            # Error for this column
            err = w_col - q_col                            # (out,)
 
            # Update W_q for remaining columns using the OBQ formula:
            #   Δw[j] = -err  *  H_inv[i, j] / H_inv[i, i]
            if i + 1 < d_in:
                h_ii = H_inv[i, i].clamp(min=1e-8)
                W_q[:, i + 1:] -= torch.outer(err, H_inv[i, i + 1:] / h_ii)
 
            W_q[:, i] = q_col
 
        self.weight.data.copy_(W_q.to(self.weight.dtype))
        
    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
 
    def forward(self, x):
        # Per-token symmetric activation quantization (same as QuantizedLinear)
        x_flat = x.reshape(-1, x.shape[-1])
        x_max = x_flat.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale_act = (x_max / self.maxq.to(x.device)).reshape(*x.shape[:-1], 1)
        maxq = self.maxq.to(x.device)
        q_act = torch.clamp(torch.round(x / scale_act), -(maxq + 1), maxq)
        x_q = (scale_act * q_act).to(x.dtype)
        # Weight is already stored quantized — no fake-quant round-trip needed
        return F.linear(x_q, self.weight, self.bias)
 
    # ------------------------------------------------------------------
    # Convenience: replace + calibrate + finalize in one call
    # ------------------------------------------------------------------
 
    @staticmethod
    def gptq_quantize_model(model, calib_loader, bits: int = 8,
                             device: torch.device = None, n_calib_batches: int = 64):
        """
        End-to-end GPTQ quantization of all nn.Linear layers in `model`.
 
        Steps:
          1. Replace every nn.Linear with GPTQLinear (hooks inactive).
          2. Run `n_calib_batches` forward passes — hooks accumulate H.
          3. Call finish_calibration() on every GPTQLinear.
 
        Args:
            model:            The nn.Module to quantize (modified in-place).
            calib_loader:     DataLoader yielding (images, labels) batches.
            bits:             Bit-width for quantization.
            device:           Device to run calibration on.
            n_calib_batches:  How many batches to use for Hessian estimation.
 
        Returns:
            model (modified in-place), dict of {name: GPTQLinear} layers.
        """
        if device is None:
            device = next(model.parameters()).device
 
        # Step 1: replace layers
        quantize_model(model, [(nn.Linear, GPTQLinear, {"bits": bits})])
        gptq_layers = find_quantized_layers(model, GPTQLinear)
 
        # Step 2: start hooks on all replaced layers
        for layer in gptq_layers.values():
            layer.start_calibration()
 
        model.eval()
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(calib_loader):
                if batch_idx >= n_calib_batches:
                    break
                model(images.to(device))
 
        # Step 3: finalize (run GPTQ, remove hooks)
        for layer in gptq_layers.values():
            layer.finish_calibration()
 
        return model, gptq_layers
 

class QuantizedConv2d(nn.Module):
    """
    Symmetric quantization for Conv2d module.
    """

    def __init__(self, original: nn.Conv2d, bits: int = 8,  asymmetric_acts: bool = False, **kwargs):
        """
        :param bits: number of bits to quantize to
        :param asymmetric_acts: False - symmetric (absmax) qaunt for both weights and inputs
                                True - symmetric (absmax) quant for the weights, asymmetric (zeropoint) quant for the inputs.
        """
        super().__init__()
        assert isinstance(original, nn.Conv2d), "QuantizedConv2d expects nn.Conv2d"
        # copy convolution characteristics
        self.in_channels = original.in_channels
        self.out_channels = original.out_channels
        self.kernel_size = original.kernel_size
        self.stride = original.stride
        self.padding = original.padding
        self.dilation = original.dilation
        self.groups = original.groups
        # quantization definitions
        self.bits = bits
        self.asymmetric_acts = asymmetric_acts
        # copy weights and bias for the original module
        self.weight = nn.Parameter(original.weight.data.clone())
        self.bias = nn.Parameter(original.bias.data.clone()) if original.bias is not None else None
        # quantization parameters (2^(b-1)-1, and save it to internal variable 'buffer').
        maxq = 2 ** (bits - 1) - 1
        self.register_buffer("maxq", torch.tensor(maxq, dtype=torch.float32))
        # compute scale factor
        self._compute_scales()

    def _compute_scales(self):
        """Compute per-channel symmetric quantization scales from the weight."""
        w = self.weight.data
        # Per-output-channel: shape (out_features,)
        w_flat = w.reshape(w.shape[0], -1)
        w_max = w_flat.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        scale = w_max / self.maxq
        self.register_buffer("scale", scale.reshape(-1, *([1] * (w.dim() - 1))))

    def _fake_quant_weight(self):
        """Quantize and dequantize the weight (fake quantization)."""
        w = self.weight
        dev = w.device
        scale = self.scale.to(dev)
        maxq = self.maxq.to(dev)
        q = torch.clamp(torch.round(w / scale), -(maxq + 1), maxq)
        return (scale * q).to(w.dtype)

    def forward(self, x):
        # check quantization type for inputs (symmetric \ asymmetric)
        if self.asymmetric_acts:
            # quant range parameters
            maxq = self.maxq.to(x.device)
            qmin = -(maxq + 1)
            q_range = (2 * maxq) + 1
            x_min = x.amin(dim=1, keepdim=True)
            x_max = x.amax(dim=1, keepdim=True)
            x_max = torch.max(x_max, x_min + 1e-8) # for the case x_min = x_max
            # calculate scale parameter and zeropoint
            scale_act = (x_max - x_min) / q_range
            z_act = torch.round(-x_min / scale_act) + qmin
            # the quantization
            q_act = torch.clamp(torch.round(x / scale_act + z_act), qmin, maxq)
            x_q = ((q_act - z_act) * scale_act).to(x.dtype)
        else:
            # 1. Online per-token activation scale
            x_max = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
            scale_act = x_max / self.maxq.to(x.device)
            # 2. Fake-quantize (symmetric) activation
            maxq = self.maxq.to(x.device)
            q_act = torch.clamp(torch.round(x / scale_act), -(maxq + 1), maxq)
            x_q = (scale_act * q_act).to(x.dtype)

        # 3. Fake-quantize (symmetric) weight and compute output
        w_q = self._fake_quant_weight()
        return F.conv2d(x_q, w_q, self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)


def quantize_model(model, replacement_list, input_quantize_list=None, name: str = ""):
    """
    Recursively replace layers according to replacement_list, and optionally
    wrap layers in input_quantize_list with InputQuantizedWrapper.

    Args:
        model: Root module to process.
        replacement_list: List of (source_type, replacement_cls) or
                         (source_type, replacement_cls, replacement_kwargs).
                         Each tuple specifies: which layer type to replace,
                         and the class to replace it with.
        input_quantize_list: Optional list of (layer_type,) or
                            (layer_type, input_quantize_kwargs). Layers matching
                            these types will be wrapped with InputQuantizedWrapper
                            so their inputs are quantized. Applied after replacement.
        name: Internal use for recursion.
    """
    replacement_clses = [r[1] for r in replacement_list]
    if any(isinstance(model, cls) for cls in replacement_clses):
        return
    if isinstance(model, InputQuantizedWrapper):
        return  # Don't recurse into wrapper to avoid double-wrapping

    # Build lookup: source_type -> (replacement_cls, kwargs)
    rules = {}
    for item in replacement_list:
        source_type = item[0]
        replacement_cls = item[1]
        replacement_kwargs = item[2] if len(item) > 2 else {}
        rules[source_type] = (replacement_cls, replacement_kwargs)

    # Build input quantize lookup: layer_type -> kwargs
    input_quantize_rules = {}
    if input_quantize_list:
        for item in input_quantize_list:
            layer_type = item[0]
            input_quantize_kwargs = item[1] if len(item) > 1 else {}
            input_quantize_rules[layer_type] = input_quantize_kwargs

    def process_module(tmp):
        """Apply replacement and/or input quantization to a module."""
        result = tmp
        if type(tmp) in rules:
            replacement_cls, replacement_kwargs = rules[type(tmp)]
            result = replacement_cls(tmp, **replacement_kwargs)
        if type(result) in input_quantize_rules and not isinstance(result, InputQuantizedWrapper):
            kwargs = input_quantize_rules[type(result)]
            result = InputQuantizedWrapper(result, **kwargs)
        return result

    for attr in dir(model):
        if attr.startswith("_"):
            continue
        try:
            tmp = getattr(model, attr)
        except (AttributeError, TypeError):
            continue
        if type(tmp) in rules or type(tmp) in input_quantize_rules:
            setattr(model, attr, process_module(tmp))
        elif isinstance(tmp, nn.Sequential):
            processed = [process_module(child) for child in tmp.children()]
            setattr(model, attr, nn.Sequential(*processed))
        elif isinstance(tmp, nn.ModuleList):
            processed = [process_module(child) for child in tmp.children()]
            setattr(model, attr, nn.ModuleList(processed))

    for child_name, child in list(model.named_children()):
        quantize_model(
            child,
            replacement_list,
            input_quantize_list=input_quantize_list,
            name=f"{name}.{child_name}" if name else child_name,
        )


def find_quantized_layers(model, replacement_cls, name: str = ""):
    """
    Recursively find all modules that are instances of replacement_cls.

    Returns:
        Dict mapping full module name to the replacement module.
    """
    result = {}
    for child_name, child in model.named_children():
        full_name = f"{name}.{child_name}" if name else child_name
        if isinstance(child, replacement_cls):
            result[full_name] = child
        else:
            result.update(find_quantized_layers(child, replacement_cls, full_name))
    return result

def quantize_depthwise_conv2d(model, bits: int = 8, asymmetric_acts: bool = False, name: str = ""):
    """
    Recursively replace only depthwise Conv2d layers with QuantizedConv2d.

    A depthwise convolution is identified by:
        isinstance(layer, nn.Conv2d) and layer.groups == layer.in_channels
    """
    for child_name, child in list(model.named_children()):
        full_name = f"{name}.{child_name}" if name else child_name

        if isinstance(child, nn.Conv2d) and child.groups == child.in_channels:
            setattr(
                model,
                child_name,
                QuantizedConv2d(child, bits=bits, asymmetric_acts=asymmetric_acts),
            )
        else:
            quantize_depthwise_conv2d(
                child,
                bits=bits,
                asymmetric_acts=asymmetric_acts,
                name=full_name,
            )


if __name__ == "__main__":
    print("Loading pre-trained ViT-B/16 from timm (ImageNet-1K)...")
    model = load_pretrained_vit("vit_base_patch16_224", pretrained=True)
    model.eval()

    print("\n--- Model architecture (before quantization) ---")
    print(model)

    print("\n--- Applying quantization to nn.Linear layers ---")
    quantize_model(model, [(nn.Linear, QuantizedLinear, {"bits": 8})])

    print("\n--- Model architecture (after quantization) ---")
    print(model)

    replaced = find_quantized_layers(model, QuantizedLinear)
    print(f"\nQuantized {len(replaced)} layers: {list(replaced.keys())[:5]}...")

    print("\n--- Running dummy forward pass ---")
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print("Forward pass succeeded.")
