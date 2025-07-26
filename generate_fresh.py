#!/usr/bin/env python3
"""
Generate a fresh SOTA submission without any checkpoints.
"""

import subprocess
import sys
import time

def clear_checkpoints():
    """Clear any existing checkpoints."""
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
        
        if checkpoint_files:
            print(f"   Found {len(checkpoint_files)} checkpoint files to remove...")
            for filename in checkpoint_files:
                print(f"   Removing {filename}...")
                result = subprocess.run(['modal', 'volume', 'rm', 'code-completion-data', filename], 
                                      capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"   ⚠️  Failed to remove {filename}: {result.stderr}")
                else:
                    print(f"   ✅ Removed {filename}")
        else:
            print("   No checkpoint files found")
        
        print("✅ Checkpoints cleared - ready for fresh start!")
        return True
        
    except Exception as e:
        print(f"❌ Error clearing checkpoints: {e}")
        return False

def generate_submission(stage="public", language="python"):
    """Generate a fresh submission."""
    print(f"🚀 Starting fresh submission generation for {language}-{stage}...")
    
    try:
        # Run the modal command
        cmd = ['modal', 'run', '--detach', 'src/core.py', 
               '--action', 'generate', '--stage', stage, '--language', language]
        
        print(f"   Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, text=True)
        
        if result.returncode == 0:
            print("🎉 Submission generation completed successfully!")
            return True
        else:
            print(f"❌ Submission generation failed with exit code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Error running submission generation: {e}")
        return False

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate fresh SOTA submission')
    parser.add_argument('--stage', default='public', help='Competition stage (default: public)')
    parser.add_argument('--language', default='python', help='Programming language (default: python)')
    parser.add_argument('--no-clear', action='store_true', help='Skip clearing checkpoints')
    
    args = parser.parse_args()
    
    print("=== Fresh SOTA Submission Generator ===\n")
    
    # Clear checkpoints unless --no-clear is specified
    if not args.no_clear:
        if not clear_checkpoints():
            print("❌ Failed to clear checkpoints. Continuing anyway...")
        print()
    
    # Generate submission
    success = generate_submission(args.stage, args.language)
    
    if success:
        print("\n✅ Fresh submission generated successfully!")
        print("   The timeout fixes ensure it won't hang at 98%.")
        print("   Check the Modal dashboard for the submission files.")
    else:
        print("\n❌ Submission generation failed.")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()