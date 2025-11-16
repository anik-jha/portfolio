#!/usr/bin/env python3
"""
Verification script for portfolio blog structure.
Checks that all required files and images exist.
"""

from pathlib import Path

def check_structure():
    base = Path(__file__).parent
    
    print("🔍 Verifying Portfolio Structure\n")
    print("=" * 60)
    
    errors = []
    warnings = []
    success = []
    
    # Check blog files
    blog_dir = base / "actual_contents" / "quantile regression"
    expected_blogs = [f"blog{i}_medium.md" for i in range(1, 6)]
    
    print("\n📝 Blog Posts:")
    for blog in expected_blogs:
        blog_path = blog_dir / blog
        if blog_path.exists():
            size_kb = blog_path.stat().st_size / 1024
            print(f"  ✓ {blog} ({size_kb:.1f} KB)")
            success.append(f"Blog: {blog}")
        else:
            print(f"  ✗ {blog} - MISSING!")
            errors.append(f"Missing blog: {blog}")
    
    # Check images
    image_dir = base / "assets"
    expected_images = [
        "ols_vs_qr_median.png",
        "qr_lines.png",
        "ols_vs_qr_flow_detailed.png",
        "pinball_loss_shapes.png"
    ]
    
    print("\n🖼️  Images:")
    for img in expected_images:
        img_path = image_dir / img
        if img_path.exists():
            size_kb = img_path.stat().st_size / 1024
            print(f"  ✓ {img} ({size_kb:.1f} KB)")
            success.append(f"Image: {img}")
        else:
            print(f"  ✗ {img} - MISSING!")
            errors.append(f"Missing image: {img}")
    
    # Check also in subdirectory
    sub_image_dir = blog_dir / "assets"
    if sub_image_dir.exists():
        print(f"\n📁 Subdirectory Images ({sub_image_dir}):")
        for img in expected_images:
            img_path = sub_image_dir / img
            if img_path.exists():
                size_kb = img_path.stat().st_size / 1024
                print(f"  ✓ {img} ({size_kb:.1f} KB)")
    
    # Check key configuration files
    print("\n⚙️  Configuration:")
    config_files = {
        "index.html": base / "index.html",
        "posts.js": base / "posts.js",
        "README.md": blog_dir / "README.md",
        "generate_images.py": sub_image_dir / "generate_images.py"
    }
    
    for name, path in config_files.items():
        if path.exists():
            size_kb = path.stat().st_size / 1024
            print(f"  ✓ {name} ({size_kb:.1f} KB)")
            success.append(f"Config: {name}")
        else:
            print(f"  ✗ {name} - MISSING!")
            warnings.append(f"Missing config: {name}")
    
    # Check virtual environment
    print("\n🐍 Virtual Environment:")
    venv_path = base / ".venv"
    if venv_path.exists():
        print(f"  ✓ .venv exists")
        
        # Check for key packages
        site_packages = venv_path / "lib"
        if site_packages.exists():
            packages = ["matplotlib", "numpy", "scipy", "sklearn"]
            for pkg in packages:
                pkg_found = any((site_packages / "python3.12" / "site-packages").glob(f"{pkg}*"))
                if pkg_found:
                    print(f"    ✓ {pkg} installed")
                else:
                    print(f"    ⚠ {pkg} may not be installed")
                    warnings.append(f"Package check: {pkg}")
    else:
        print(f"  ⚠ .venv not found")
        warnings.append("Virtual environment not set up")
    
    # Summary
    print("\n" + "=" * 60)
    print(f"\n📊 Summary:")
    print(f"  ✓ Success: {len(success)} items")
    print(f"  ⚠ Warnings: {len(warnings)} items")
    print(f"  ✗ Errors: {len(errors)} items")
    
    if errors:
        print("\n❌ ERRORS:")
        for err in errors:
            print(f"  - {err}")
    
    if warnings:
        print("\n⚠️  WARNINGS:")
        for warn in warnings:
            print(f"  - {warn}")
    
    if not errors and not warnings:
        print("\n✅ All checks passed! Portfolio structure is complete.")
        return 0
    elif not errors:
        print("\n⚠️  Structure is functional but has warnings.")
        return 0
    else:
        print("\n❌ Critical errors found. Please fix before deployment.")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(check_structure())
