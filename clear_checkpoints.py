#!/usr/bin/env python3
"""
Clear any existing checkpoints to ensure fresh starts.
"""

import subprocess
import sys

def clear_checkpoints():
    """Clear all checkpoint files from Modal volume."""
    print("🧹 Clearing checkpoints for fresh start...")
    
    try:
        # List files in volume to see what checkpoints exist
        result = subprocess.run(['modal', 'volume', 'ls', 'code-completion-data'], 
                              capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Failed to list volume contents: {result.stderr}")
            return False
        
        # Look for checkpoint files
        checkpoint_files = []
        for line in result.stdout.split('\n'):
            if 'checkpoint.json' in line:
                # Extract filename from the line
                parts = line.split()
                if parts:
                    filename = parts[0]
                    checkpoint_files.append(filename)
        
        if not checkpoint_files:
            print("✅ No checkpoint files found - ready for fresh start!")
            return True
        
        # Remove checkpoint files
        for filename in checkpoint_files:
            print(f"   Removing {filename}...")
            result = subprocess.run(['modal', 'volume', 'rm', 'code-completion-data', filename], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print(f"   ⚠️  Failed to remove {filename}: {result.stderr}")
            else:
                print(f"   ✅ Removed {filename}")
        
        print("🎉 All checkpoints cleared - ready for fresh start!")
        return True
        
    except Exception as e:
        print(f"❌ Error clearing checkpoints: {e}")
        return False

if __name__ == "__main__":
    success = clear_checkpoints()
    sys.exit(0 if success else 1)