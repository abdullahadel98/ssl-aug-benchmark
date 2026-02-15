#!/usr/bin/env python3
"""
Comprehensive test script for style_transfer.py module.

Tests:
1. Model loading (VGG encoder, decoder)
2. Style features loading
3. NSTTransform input/output validation
4. Batch augmentation integration
5. Device handling (CPU/GPU)
6. Shape and dtype consistency
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Tuple, List

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "learning" / "solo-learn"))

# Conditional imports based on availability
try:
    from solo.data.style_transfer import load_models, load_feat_files, NSTTransform
    STYLE_TRANSFER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import style_transfer modules: {e}")
    STYLE_TRANSFER_AVAILABLE = False

try:
    from solo.methods.batch_augmentation_mixin import BatchAugmentationMixin
    MIXIN_AVAILABLE = True
except ImportError:
    MIXIN_AVAILABLE = False


class StyleTransferTester:
    """Comprehensive tester for style transfer augmentation."""
    
    def __init__(self, device: str = "cpu"):
        """
        Initialize tester.
        
        Args:
            device: "cpu", "cuda", or "cuda:0"
        """
        self.device = torch.device(device)
        self.results = {}
        self.test_count = 0
        self.passed_count = 0
        
        print(f"\n{'='*80}")
        print(f"StyleTransfer Module Test Suite")
        print(f"Device: {self.device}")
        print(f"{'='*80}\n")
    
    def log_test(self, name: str, passed: bool, message: str = ""):
        """Log test result."""
        self.test_count += 1
        if passed:
            self.passed_count += 1
            status = "✓ PASS"
        else:
            status = "✗ FAIL"
        
        print(f"[{status}] {name}")
        if message:
            print(f"      {message}")
        self.results[name] = passed
    
    def test_synthetic_style_features(self):
        """Test with synthetic style features (no external files needed)."""
        print(f"\n{'─'*80}")
        print("Test 1: Synthetic Style Features Creation")
        print(f"{'─'*80}")
        
        try:
            # Create synthetic style features matching expected shape [N, C]
            # Typical: [1000, 512] for 1000 style images with 512-dim AdaIN features
            style_feats = torch.randn(10, 512, device=self.device)
            
            self.log_test(
                "Create synthetic style features",
                True,
                f"Shape: {tuple(style_feats.shape)}, dtype: {style_feats.dtype}"
            )
            return style_feats
        except Exception as e:
            self.log_test("Create synthetic style features", False, str(e))
            return None
    
    def test_synthetic_models(self) -> Tuple[nn.Module, nn.Module]:
        """Create synthetic VGG and decoder for testing."""
        print(f"\n{'─'*80}")
        print("Test 2: Synthetic Model Creation (Mock VGG + Decoder)")
        print(f"{'─'*80}")
        
        try:
            # Mock VGG encoder (simplified version)
            class MockVGG(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.layers = nn.Sequential(
                        nn.Conv2d(3, 64, 3, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(64, 512, 3, padding=1),
                    )
                
                def forward(self, x):
                    return self.layers(x)
            
            # Mock Decoder
            class MockDecoder(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.layers = nn.Sequential(
                        nn.Conv2d(512, 256, 3, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(256, 3, 3, padding=1),
                    )
                
                def forward(self, x):
                    return torch.clamp(self.layers(x), 0, 1)
            
            vgg = MockVGG().to(self.device).eval()
            decoder = MockDecoder().to(self.device).eval()
            
            # Disable gradients
            for param in vgg.parameters():
                param.requires_grad = False
            for param in decoder.parameters():
                param.requires_grad = False
            
            self.log_test(
                "Create synthetic VGG encoder",
                True,
                f"Total params: {sum(p.numel() for p in vgg.parameters())}"
            )
            self.log_test(
                "Create synthetic Decoder",
                True,
                f"Total params: {sum(p.numel() for p in decoder.parameters())}"
            )
            
            return vgg, decoder
        except Exception as e:
            self.log_test("Create synthetic models", False, str(e))
            return None, None
    
    def test_nst_transform_init(self, style_feats, vgg, decoder):
        """Test NSTTransform initialization."""
        print(f"\n{'─'*80}")
        print("Test 3: NSTTransform Initialization")
        print(f"{'─'*80}")
        
        try:
            if not STYLE_TRANSFER_AVAILABLE:
                self.log_test("Initialize NSTTransform", False, "style_transfer module not available")
                return None
            
            transform = NSTTransform(
                style_feats=style_feats,
                vgg=vgg,
                decoder=decoder,
                alpha_min=0.7,
                alpha_max=1.0,
                probability=0.5,
                device=self.device
            )
            
            self.log_test(
                "Initialize NSTTransform",
                True,
                f"Num styles: {transform.num_styles}, Probability: {transform.probability}"
            )
            return transform
        except Exception as e:
            self.log_test("Initialize NSTTransform", False, str(e))
            return None
    
    def test_single_image_input(self, transform):
        """Test with single image [C, H, W]."""
        print(f"\n{'─'*80}")
        print("Test 4: Single Image Input [C, H, W]")
        print(f"{'─'*80}")
        
        try:
            if transform is None:
                self.log_test("Single image (3, 32, 32)", False, "Transform not initialized")
                return
            
            # Test different image sizes
            test_sizes = [(3, 32, 32), (3, 64, 64), (1, 32, 32)]
            
            for shape in test_sizes:
                img = torch.rand(shape, device=self.device)
                
                output = transform(img)
                
                # Validate output - normalize device comparison
                output_device = output.device.type if hasattr(output.device, 'type') else str(output.device)
                self_device = self.device.type if hasattr(self.device, 'type') else str(self.device)
                
                checks = [
                    (output.shape == img.shape, f"Shape match: {output.shape} == {img.shape}"),
                    (output.dtype == img.dtype, f"Dtype match: {output.dtype} == {img.dtype}"),
                    (output_device == self_device, f"Device match: {output_device} == {self_device}"),
                    (torch.isfinite(output).all(), "All values finite"),
                ]
                
                all_passed = all(check[0] for check in checks)
                self.log_test(
                    f"Single image {shape}",
                    all_passed,
                    " | ".join(f"{check[1]}: {check[0]}" for check in checks)
                )
        except Exception as e:
            self.log_test("Single image input", False, str(e))
    
    def test_batch_input(self, transform):
        """Test with batch input [B, C, H, W]."""
        print(f"\n{'─'*80}")
        print("Test 5: Batch Input [B, C, H, W]")
        print(f"{'─'*80}")
        
        try:
            if transform is None:
                self.log_test("Batch input", False, "Transform not initialized")
                return
            
            # Test different batch sizes and image sizes
            test_configs = [
                (2, 3, 32, 32),    # Small batch, small image
                (8, 3, 64, 64),    # Medium batch, medium image
                (16, 3, 224, 224), # Large batch, standard size
                (4, 1, 32, 32),    # Grayscale batch
            ]
            
            for shape in test_configs:
                try:
                    batch = torch.rand(shape, device=self.device)
                    
                    output = transform(batch)
                    
                    # Validate output
                    checks = [
                        (output.shape == batch.shape, f"Shape: {output.shape} == {batch.shape}"),
                        (output.dtype == batch.dtype, f"Dtype: {output.dtype} == {batch.dtype}"),
                        (0 <= output.min() and output.max() <= 1, 
                         f"Value range: [{output.min():.3f}, {output.max():.3f}]"),
                        (torch.isfinite(output).all(), "All values finite"),
                    ]
                    
                    all_passed = all(check[0] for check in checks)
                    self.log_test(
                        f"Batch {shape}",
                        all_passed,
                        " | ".join(f"{check[1]}: {check[0]}" for check in checks)
                    )
                except Exception as e:
                    self.log_test(f"Batch {shape}", False, str(e))
        except Exception as e:
            self.log_test("Batch input (outer)", False, str(e))
    
    def test_different_dtypes(self, transform):
        """Test with different data types (float32, float64, bfloat16)."""
        print(f"\n{'─'*80}")
        print("Test 6: Different Data Types")
        print(f"{'─'*80}")
        
        try:
            if transform is None:
                self.log_test("Different dtypes", False, "Transform not initialized")
                return
            
            dtypes_to_test = [torch.float32, torch.float64]
            
            # Add bfloat16 if CUDA is available
            if torch.cuda.is_available():
                dtypes_to_test.append(torch.bfloat16)
            
            for dtype in dtypes_to_test:
                try:
                    batch = torch.rand(4, 3, 32, 32, dtype=dtype, device=self.device)
                    output = transform(batch)
                    
                    # Check if output maintains original dtype or converts to float32
                    # (which is acceptable since VGG requires float32)
                    passed = output.dtype in [dtype, torch.float32]
                    message = f"Input dtype: {dtype} → Output dtype: {output.dtype}"
                    
                    self.log_test(f"Dtype {dtype}", passed, message)
                except Exception as e:
                    self.log_test(f"Dtype {dtype}", False, str(e))
        except Exception as e:
            self.log_test("Different dtypes test", False, str(e))
    
    def test_probability_application(self, transform):
        """Test that style transfer is applied with correct probability."""
        print(f"\n{'─'*80}")
        print("Test 7: Probability-based Application")
        print(f"{'─'*80}")
        
        try:
            if transform is None:
                self.log_test("Probability test", False, "Transform not initialized")
                return
            
            # Create batch with high probability
            original_prob = transform.probability
            transform.probability = 1.0  # Always apply
            
            batch_original = torch.rand(4, 3, 32, 32, device=self.device)
            batch_augmented = transform(batch_original)
            
            # With probability=1.0, output should differ from input
            is_different = not torch.allclose(batch_original, batch_augmented, atol=1e-5)
            self.log_test(
                "Probability = 1.0 (always apply)",
                is_different,
                f"Output differs from input: {is_different}"
            )
            
            # Test with probability = 0.0
            transform.probability = 0.0  # Never apply
            batch_no_aug = transform(batch_original)
            
            is_same = torch.allclose(batch_original, batch_no_aug, atol=1e-5)
            self.log_test(
                "Probability = 0.0 (never apply)",
                is_same,
                f"Output same as input: {is_same}"
            )
            
            # Restore original probability
            transform.probability = original_prob
        except Exception as e:
            self.log_test("Probability test", False, str(e))
    
    def test_alpha_blending(self, transform):
        """Test alpha blending strength effect."""
        print(f"\n{'─'*80}")
        print("Test 8: Alpha Blending Strength")
        print(f"{'─'*80}")
        
        try:
            if transform is None:
                self.log_test("Alpha blending test", False, "Transform not initialized")
                return
            
            batch = torch.rand(4, 3, 32, 32, device=self.device)
            
            # Test alpha_min = alpha_max (fixed alpha)
            original_alpha_min, original_alpha_max = transform.alpha_min, transform.alpha_max
            
            # Full style transfer (alpha=1.0)
            transform.alpha_min = 1.0
            transform.alpha_max = 1.0
            transform.probability = 1.0
            output_full = transform(batch.clone())
            
            # Partial style transfer (alpha=0.5)
            transform.alpha_min = 0.5
            transform.alpha_max = 0.5
            output_half = transform(batch.clone())
            
            # No style transfer (alpha=0.0)
            transform.alpha_min = 0.0
            transform.alpha_max = 0.0
            output_none = transform(batch.clone())
            
            # Outputs should differ
            full_vs_half = not torch.allclose(output_full, output_half, atol=1e-2)
            half_vs_none = not torch.allclose(output_half, output_none, atol=1e-2)
            
            self.log_test(
                "Alpha=1.0 vs Alpha=0.5 produces different outputs",
                full_vs_half,
                f"Difference: {(output_full - output_half).abs().mean():.6f}"
            )
            
            self.log_test(
                "Alpha=0.5 vs Alpha=0.0 produces different outputs",
                half_vs_none,
                f"Difference: {(output_half - output_none).abs().mean():.6f}"
            )
            
            # Restore original values
            transform.alpha_min = original_alpha_min
            transform.alpha_max = original_alpha_max
        except Exception as e:
            self.log_test("Alpha blending test", False, str(e))
    
    def test_grayscale_handling(self, transform):
        """Test grayscale image handling."""
        print(f"\n{'─'*80}")
        print("Test 9: Grayscale Image Handling")
        print(f"{'─'*80}")
        
        try:
            if transform is None:
                self.log_test("Grayscale test", False, "Transform not initialized")
                return
            
            # Single grayscale image
            gray_single = torch.rand(1, 32, 32, device=self.device)
            output_single = transform(gray_single)
            
            self.log_test(
                "Single grayscale image [1, H, W]",
                output_single.shape == gray_single.shape,
                f"Shape: {output_single.shape} == {gray_single.shape}"
            )
            
            # Batch of grayscale images
            gray_batch = torch.rand(4, 1, 32, 32, device=self.device)
            output_batch = transform(gray_batch)
            
            self.log_test(
                "Batch grayscale images [B, 1, H, W]",
                output_batch.shape == gray_batch.shape,
                f"Shape: {output_batch.shape} == {gray_batch.shape}"
            )
        except Exception as e:
            self.log_test("Grayscale handling", False, str(e))
    
    def test_memory_efficiency(self, transform):
        """Test memory usage with large batches."""
        print(f"\n{'─'*80}")
        print("Test 10: Memory Efficiency")
        print(f"{'─'*80}")
        
        try:
            if transform is None:
                self.log_test("Memory test", False, "Transform not initialized")
                return
            
            # Get initial memory
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                mem_before = torch.cuda.memory_allocated() / 1e6  # MB
            
            # Process batch
            batch = torch.rand(16, 3, 64, 64, device=self.device)
            output = transform(batch)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                mem_after = torch.cuda.memory_allocated() / 1e6  # MB
                mem_used = mem_after - mem_before
                
                self.log_test(
                    "Large batch memory usage",
                    mem_used < 1000,  # Less than 1GB
                    f"Memory used: {mem_used:.1f} MB"
                )
            else:
                self.log_test(
                    "Large batch memory usage",
                    True,
                    "CPU mode - skipping GPU memory check"
                )
        except Exception as e:
            self.log_test("Memory test", False, str(e))
    
    def test_device_transfers(self, transform):
        """Test device transfer (CPU ↔ GPU)."""
        print(f"\n{'─'*80}")
        print("Test 11: Device Transfer")
        print(f"{'─'*80}")
        
        try:
            if transform is None:
                self.log_test("Device transfer test", False, "Transform not initialized")
                return
            
            batch = torch.rand(4, 3, 32, 32)
            
            # Test on CPU
            batch_cpu = batch.to("cpu")
            transform.device = torch.device("cpu")
            # Also move style features to CPU
            if hasattr(transform, 'style_features'):
                transform.style_features = transform.style_features.to("cpu")
            output_cpu = transform(batch_cpu)
            
            self.log_test(
                "CPU processing",
                output_cpu.device.type == "cpu",
                f"Output device: {output_cpu.device}"
            )
            
            # Test on GPU if available
            if torch.cuda.is_available():
                batch_gpu = batch.to("cuda")
                transform.device = torch.device("cuda")
                # Also move style features to GPU
                if hasattr(transform, 'style_features'):
                    transform.style_features = transform.style_features.to("cuda")
                output_gpu = transform(batch_gpu)
                
                self.log_test(
                    "GPU processing",
                    output_gpu.device.type == "cuda",
                    f"Output device: {output_gpu.device}"
                )
        except Exception as e:
            self.log_test("Device transfer", False, str(e))
    
    def test_no_grad_mode(self, transform):
        """Test that augmentation works in no_grad mode."""
        print(f"\n{'─'*80}")
        print("Test 12: No-Gradient Mode (Inference)")
        print(f"{'─'*80}")
        
        try:
            if transform is None:
                self.log_test("No-grad mode test", False, "Transform not initialized")
                return
            
            batch = torch.rand(4, 3, 32, 32, device=self.device, requires_grad=True)
            
            with torch.no_grad():
                output = transform(batch)
            
            # Output should not require gradients
            self.log_test(
                "Output in no_grad mode",
                not output.requires_grad,
                f"Requires grad: {output.requires_grad}"
            )
            
            self.log_test(
                "Output shape preserved",
                output.shape == batch.shape,
                f"Shape: {output.shape}"
            )
        except Exception as e:
            self.log_test("No-grad mode test", False, str(e))
    
    def test_reproducibility(self, transform):
        """Test reproducibility with fixed random seed."""
        print(f"\n{'─'*80}")
        print("Test 13: Reproducibility with Random Seed")
        print(f"{'─'*80}")
        
        try:
            if transform is None:
                self.log_test("Reproducibility test", False, "Transform not initialized")
                return
            
            batch = torch.rand(4, 3, 32, 32, device=self.device)
            
            # First run
            torch.manual_seed(42)
            np.random.seed(42)
            output1 = transform(batch.clone())
            
            # Second run with same seed
            torch.manual_seed(42)
            np.random.seed(42)
            output2 = transform(batch.clone())
            
            is_reproducible = torch.allclose(output1, output2, atol=1e-5)
            self.log_test(
                "Reproducible with fixed seed",
                is_reproducible,
                f"Max difference: {(output1 - output2).abs().max():.2e}"
            )
        except Exception as e:
            self.log_test("Reproducibility test", False, str(e))
    
    def print_summary(self):
        """Print test summary."""
        print(f"\n{'='*80}")
        print("Test Summary")
        print(f"{'='*80}")
        print(f"Total Tests: {self.test_count}")
        print(f"Passed: {self.passed_count}")
        print(f"Failed: {self.test_count - self.passed_count}")
        print(f"Success Rate: {100 * self.passed_count / max(self.test_count, 1):.1f}%")
        print(f"{'='*80}\n")
        
        return self.passed_count == self.test_count
    
    def run_all_tests(self):
        """Run all tests."""
        # Create synthetic data
        style_feats = self.test_synthetic_style_features()
        vgg, decoder = self.test_synthetic_models()
        
        # Test initialization
        transform = self.test_nst_transform_init(style_feats, vgg, decoder)
        
        if transform is not None:
            # Test input/output
            self.test_single_image_input(transform)
            self.test_batch_input(transform)
            self.test_different_dtypes(transform)
            
            # Test functionality
            self.test_probability_application(transform)
            self.test_alpha_blending(transform)
            self.test_grayscale_handling(transform)
            
            # Test robustness
            self.test_memory_efficiency(transform)
            self.test_device_transfers(transform)
            self.test_no_grad_mode(transform)
            self.test_reproducibility(transform)
        
        # Print results
        return self.print_summary()


def test_batch_augmentation_mixin():
    """Test batch augmentation mixin integration."""
    print(f"\n{'='*80}")
    print("Batch Augmentation Mixin Integration Test")
    print(f"{'='*80}\n")
    
    if not MIXIN_AVAILABLE:
        print("✗ BatchAugmentationMixin not available - skipping integration test")
        return True
    
    try:
        # Create minimal test class
        class MockSSLMethod(BatchAugmentationMixin):
            def __init__(self, device):
                self.device = device
                self.batch_augmentations = {}
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MockSSLMethod(device)
        
        # Create mock config
        from omegaconf import DictConfig, OmegaConf
        
        cfg = OmegaConf.create({
            "batch_augmentations": {
                "style_transfer": {
                    "enabled": False  # Disabled to avoid file loading
                }
            }
        })
        
        # Test setup
        model.setup_batch_augmentations(cfg)
        print("✓ BatchAugmentationMixin setup successful")
        
        # Test apply_batch_augmentations
        X = [torch.rand(4, 3, 32, 32, device=device)]
        X_augmented = model.apply_batch_augmentations(X)
        
        print(f"✓ apply_batch_augmentations works: {len(X_augmented)} crops")
        
        # Test statistics
        stats = model.get_batch_augmentation_stats()
        print(f"✓ Augmentation stats: {stats}")
        
        return True
    except Exception as e:
        print(f"✗ Mixin integration test failed: {e}")
        return False


def main():
    """Main test entry point."""
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Run main tests
    tester = StyleTransferTester(device=device)
    style_transfer_passed = tester.run_all_tests()
    
    # Run integration test
    mixin_passed = test_batch_augmentation_mixin()
    
    # Overall result
    print(f"\n{'='*80}")
    print("Overall Test Results")
    print(f"{'='*80}")
    print(f"Style Transfer Tests: {'✓ PASSED' if style_transfer_passed else '✗ FAILED'}")
    print(f"Mixin Integration Test: {'✓ PASSED' if mixin_passed else '✗ FAILED'}")
    print(f"{'='*80}\n")
    
    return 0 if (style_transfer_passed and mixin_passed) else 1


if __name__ == "__main__":
    exit(main())
