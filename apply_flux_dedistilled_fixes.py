#!/usr/bin/env python3
"""
Flux Dedistilled Fixes for OneTrainer
======================================

This script automatically applies fixes to enable Flux Dev Dedistilled model support in OneTrainer.

Fixes applied:
1. Flux Dedistilled crash prevention (None-safe guidance parameter)
2. Traditional CFG support for dedistilled models during sampling
3. Guidance scale parameter copying for sample generation

Usage:
    python apply_flux_dedistilled_fixes.py

Or on Windows:
    apply_flux_dedistilled_fixes.bat
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import Tuple, List, Dict

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text: str):
    """Print a styled header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text:^70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}\n")

def print_success(text: str):
    """Print success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_error(text: str):
    """Print error message."""
    print(f"{Colors.RED}✗ {text}{Colors.END}")

def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")

def print_info(text: str):
    """Print info message."""
    print(f"{Colors.BLUE}ℹ {text}{Colors.END}")

# File paths
FILES_TO_MODIFY = {
    'FluxSampler': 'modules/modelSampler/FluxSampler.py',
    'BaseFluxSetup': 'modules/modelSetup/BaseFluxSetup.py',
    'SampleConfig': 'modules/util/config/SampleConfig.py'
}

PATCH_FILE = 'flux-dedistilled-fixes.patch'

def check_files_exist() -> Tuple[bool, List[str]]:
    """Check if all required files exist."""
    missing = []
    for name, path in FILES_TO_MODIFY.items():
        if not os.path.exists(path):
            missing.append(path)
    return len(missing) == 0, missing

def create_backups() -> bool:
    """Create backup copies of files before modification."""
    print_info("Creating backups...")
    try:
        for name, path in FILES_TO_MODIFY.items():
            backup_path = f"{path}.bak"
            shutil.copy2(path, backup_path)
            print_success(f"Backed up: {path} → {backup_path}")
        return True
    except Exception as e:
        print_error(f"Failed to create backups: {e}")
        return False

def restore_backups():
    """Restore files from backups."""
    print_warning("Restoring from backups...")
    for name, path in FILES_TO_MODIFY.items():
        backup_path = f"{path}.bak"
        if os.path.exists(backup_path):
            shutil.copy2(backup_path, path)
            print_info(f"Restored: {backup_path} → {path}")

def try_apply_patch() -> bool:
    """Try to apply the git patch file."""
    if not os.path.exists(PATCH_FILE):
        print_warning(f"Patch file not found: {PATCH_FILE}")
        return False
    
    print_info("Attempting to apply git patch...")
    try:
        # Try git apply first
        result = subprocess.run(
            ['git', 'apply', '--check', PATCH_FILE],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            # Patch can be applied cleanly
            result = subprocess.run(
                ['git', 'apply', PATCH_FILE],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print_success("Git patch applied successfully!")
                return True
            else:
                print_warning(f"Git apply failed: {result.stderr}")
                return False
        else:
            print_warning("Patch cannot be applied cleanly (conflicts detected)")
            print_info("Falling back to manual fixes...")
            return False
            
    except FileNotFoundError:
        print_warning("Git not found in PATH, falling back to manual fixes...")
        return False
    except Exception as e:
        print_warning(f"Error applying patch: {e}")
        return False

def apply_manual_fixes() -> Dict[str, bool]:
    """Apply fixes manually using find/replace."""
    results = {}
    
    print_info("Applying manual fixes...")
    
    # Fix 1 & 2: FluxSampler.py - Guidance None check and CFG implementation
    results['FluxSampler'] = apply_flux_sampler_fixes()
    
    # Fix 3: BaseFluxSetup.py - Guidance None check
    results['BaseFluxSetup'] = apply_base_flux_setup_fix()
    
    # Fix 4: SampleConfig.py - Guidance scale copy
    results['SampleConfig'] = apply_sample_config_fix()
    
    return results

def apply_flux_sampler_fixes() -> bool:
    """Apply fixes to FluxSampler.py."""
    filepath = FILES_TO_MODIFY['FluxSampler']
    print_info(f"Fixing {filepath}...")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        fixes_applied = 0
        
        # Fix 1: Add CFG detection and negative prompt encoding (before denoising loop in __sample_base)
        if 'use_cfg = not transformer.config.guidance_embeds and cfg_scale > 1.0' not in content:
            # Find the location after positive prompt encoding
            search_str = '''            )

            self.model.text_encoder_to(self.temp_device)'''
            
            if search_str in content:
                replace_str = '''            )

            # encode negative prompt for CFG (dedistilled models)
            use_cfg = not transformer.config.guidance_embeds and cfg_scale > 1.0
            if use_cfg:
                negative_prompt_embedding, negative_pooled_prompt_embedding = self.model.encode_text(
                    text=negative_prompt if negative_prompt else "",
                    train_device=self.train_device,
                    text_encoder_1_layer_skip=text_encoder_1_layer_skip,
                    text_encoder_2_layer_skip=text_encoder_2_layer_skip,
                    apply_attention_mask=prior_attention_mask,
                )

            self.model.text_encoder_to(self.temp_device)'''
                
                content = content.replace(search_str, replace_str, 1)
                fixes_applied += 1
                print_success("  Added CFG detection and negative prompt encoding")
        
        # Fix 2: Replace single-pass transformer call with CFG-aware two-pass in __sample_base
        # This is complex, so we check if it's already applied
        if 'if use_cfg:' in content and 'Flux Dedistilled with CFG: two-pass approach' in content:
            print_success("  CFG implementation already present")
            fixes_applied += 1
        else:
            # Need to implement the full CFG logic - this is the most complex fix
            # Look for the guidance handling section
            old_pattern = '''                # predict the noise residual
                noise_pred = transformer(
                    hidden_states=latent_model_input.to(dtype=self.model.train_dtype.torch_dtype()),
                    timestep=expanded_timestep / 1000,
                    guidance=guidance.to(dtype=self.model.train_dtype.torch_dtype()),'''
            
            if old_pattern in content:
                # This means fixes haven't been applied yet
                print_error("  Complex CFG implementation needed - patch file required!")
                return False
        
        # Fix 3: Ensure guidance None-safe in both methods
        old_guidance = 'guidance=guidance.to(dtype=self.model.train_dtype.torch_dtype()),'
        new_guidance = 'guidance=guidance.to(dtype=self.model.train_dtype.torch_dtype()) if guidance is not None else None,'
        
        if old_guidance in content and new_guidance not in content:
            # Count occurrences and replace all
            count = content.count(old_guidance)
            content = content.replace(old_guidance, new_guidance)
            fixes_applied += count
            print_success(f"  Applied guidance None-check ({count} locations)")
        elif new_guidance in content:
            print_success("  Guidance None-check already applied")
            fixes_applied += 1
        
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print_success(f"✓ FluxSampler.py: {fixes_applied} fixes applied")
            return True
        elif fixes_applied > 0:
            print_success(f"✓ FluxSampler.py: All fixes already present")
            return True
        else:
            print_error("✗ FluxSampler.py: No fixes could be applied")
            return False
            
    except Exception as e:
        print_error(f"✗ FluxSampler.py: {e}")
        return False

def apply_base_flux_setup_fix() -> bool:
    """Apply fixes to BaseFluxSetup.py."""
    filepath = FILES_TO_MODIFY['BaseFluxSetup']
    print_info(f"Fixing {filepath}...")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Fix: Guidance None check
        old_guidance = 'guidance=guidance.to(dtype=model.train_dtype.torch_dtype()),'
        new_guidance = 'guidance=guidance.to(dtype=model.train_dtype.torch_dtype()) if guidance is not None else None,'
        
        if old_guidance in content and new_guidance not in content:
            content = content.replace(old_guidance, new_guidance)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print_success("✓ BaseFluxSetup.py: Guidance None-check applied")
            return True
        elif new_guidance in content:
            print_success("✓ BaseFluxSetup.py: Guidance None-check already present")
            return True
        else:
            print_warning("✗ BaseFluxSetup.py: Pattern not found (might be already fixed)")
            return True  # Assume it's okay if pattern isn't found
            
    except Exception as e:
        print_error(f"✗ BaseFluxSetup.py: {e}")
        return False

def apply_sample_config_fix() -> bool:
    """Apply fixes to SampleConfig.py."""
    filepath = FILES_TO_MODIFY['SampleConfig']
    print_info(f"Fixing {filepath}...")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Fix: Add cfg_scale copy in from_train_config
        if 'self.cfg_scale = train_config.prior.guidance_scale' in content:
            print_success("✓ SampleConfig.py: Guidance scale copy already present")
            return True
        
        # Find the from_train_config method and add the line
        search_pattern = '''        self.text_encoder_4_layer_skip = train_config.text_encoder_4_layer_skip
        self.prior_attention_mask = train_config.prior.attention_mask
        self.force_last_timestep = train_config.rescale_noise_scheduler_to_zero_terminal_snr'''
        
        if search_pattern in content:
            replace_pattern = '''        self.text_encoder_4_layer_skip = train_config.text_encoder_4_layer_skip
        self.prior_attention_mask = train_config.prior.attention_mask
        self.force_last_timestep = train_config.rescale_noise_scheduler_to_zero_terminal_snr
        self.cfg_scale = train_config.prior.guidance_scale'''
            
            content = content.replace(search_pattern, replace_pattern)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print_success("✓ SampleConfig.py: Guidance scale copy added")
            return True
        else:
            print_warning("✗ SampleConfig.py: Pattern not found (file might have changed)")
            return False
            
    except Exception as e:
        print_error(f"✗ SampleConfig.py: {e}")
        return False

def verify_fixes() -> Tuple[bool, List[str]]:
    """Verify that all fixes were applied correctly."""
    print_info("Verifying fixes...")
    
    issues = []
    
    # Check FluxSampler.py
    try:
        with open(FILES_TO_MODIFY['FluxSampler'], 'r', encoding='utf-8') as f:
            content = f.read()
        
        checks = [
            ('use_cfg = not transformer.config.guidance_embeds', 'CFG detection'),
            ('if use_cfg:', 'CFG implementation'),
            ('guidance.to(dtype=self.model.train_dtype.torch_dtype()) if guidance is not None else None', 'Guidance None-check'),
        ]
        
        for pattern, name in checks:
            if pattern not in content:
                issues.append(f"FluxSampler.py: Missing {name}")
        
    except Exception as e:
        issues.append(f"FluxSampler.py: Could not verify - {e}")
    
    # Check BaseFluxSetup.py
    try:
        with open(FILES_TO_MODIFY['BaseFluxSetup'], 'r', encoding='utf-8') as f:
            content = f.read()
        
        if 'guidance.to(dtype=model.train_dtype.torch_dtype()) if guidance is not None else None' not in content:
            issues.append("BaseFluxSetup.py: Missing guidance None-check")
            
    except Exception as e:
        issues.append(f"BaseFluxSetup.py: Could not verify - {e}")
    
    # Check SampleConfig.py
    try:
        with open(FILES_TO_MODIFY['SampleConfig'], 'r', encoding='utf-8') as f:
            content = f.read()
        
        if 'self.cfg_scale = train_config.prior.guidance_scale' not in content:
            issues.append("SampleConfig.py: Missing guidance scale copy")
            
    except Exception as e:
        issues.append(f"SampleConfig.py: Could not verify - {e}")
    
    return len(issues) == 0, issues

def main():
    """Main execution function."""
    print_header("Flux Dedistilled Fixes for OneTrainer")
    
    # Check if we're in the right directory
    if not os.path.exists('modules'):
        print_error("Error: 'modules' directory not found!")
        print_error("Please run this script from the OneTrainer root directory.")
        return 1
    
    # Check if files exist
    files_exist, missing = check_files_exist()
    if not files_exist:
        print_error("Error: Required files not found:")
        for path in missing:
            print_error(f"  - {path}")
        return 1
    
    print_success("All required files found")
    
    # Create backups
    if not create_backups():
        return 1
    
    # Try to apply patch first
    patch_success = try_apply_patch()
    
    if not patch_success:
        # Fall back to manual fixes
        print_info("\nApplying manual fixes (this may take a moment)...")
        results = apply_manual_fixes()
        
        # Check results
        failed = [name for name, success in results.items() if not success]
        if failed:
            print_error(f"\nFailed to apply fixes to: {', '.join(failed)}")
            print_warning("Attempting to restore backups...")
            restore_backups()
            return 1
    
    # Verify fixes
    verification_success, issues = verify_fixes()
    
    if verification_success:
        print_success("\n" + "="*70)
        print_success("ALL FIXES APPLIED SUCCESSFULLY!")
        print_success("="*70)
        print_info("\nWhat was fixed:")
        print_info("  ✓ Flux Dedistilled crash prevention (guidance None-check)")
        print_info("  ✓ Traditional CFG support for dedistilled models")
        print_info("  ✓ Guidance scale parameter copying")
        print_info("\nYou can now train Flux Dev Dedistilled models!")
        print_info("\nBackup files (.bak) have been created. You can delete them once")
        print_info("you've confirmed everything works correctly.")
        return 0
    else:
        print_error("\n" + "="*70)
        print_error("VERIFICATION FAILED - SOME FIXES MAY BE INCOMPLETE")
        print_error("="*70)
        print_error("\nIssues found:")
        for issue in issues:
            print_error(f"  - {issue}")
        print_warning("\nYou may need to apply fixes manually.")
        print_warning("See FLUX_DEDISTILLED_FIXES.md for details.")
        return 1

if __name__ == '__main__':
    try:
        exit_code = main()
        print()  # Empty line before exit
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print_warning("\n\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

