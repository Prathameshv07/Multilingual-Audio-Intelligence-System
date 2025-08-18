#!/usr/bin/env python3
import re
import os
import sys
import subprocess
from pathlib import Path

def process_mermaid_diagrams(content, file_dir):
    """Convert mermaid diagrams to images"""
    mermaid_pattern = r'```mermaid\n(.*?)\n```'

    def replace_mermaid(match):
        mermaid_code = match.group(1)
        # Create a unique filename for this diagram
        diagram_hash = str(abs(hash(mermaid_code)))
        mermaid_file = f"{file_dir}/mermaid_{diagram_hash}.mmd"
        svg_file = f"{file_dir}/mermaid_{diagram_hash}.svg"
        png_file = f"{file_dir}/mermaid_{diagram_hash}.png"

        # Write mermaid code to file
        try:
            with open(mermaid_file, 'w', encoding='utf-8') as f:
                f.write(mermaid_code)
        except Exception as e:
            print(f"Error writing mermaid file: {e}")
            return f'\n```\n{mermaid_code}\n```\n'

        try:
            # Convert to SVG first - FIXED: Remove --puppeteerConfig
            result = subprocess.run([
                'mmdc', '-i', mermaid_file, '-o', svg_file,
                '--theme', 'default', '--backgroundColor', 'white'
            ], check=True, capture_output=True, text=True)

            # Convert SVG to PNG for better PDF compatibility
            subprocess.run([
                'rsvg-convert', '-f', 'png', '-o', png_file,
                '--width', '1200', '--height', '800', svg_file
            ], check=True, capture_output=True, text=True)

            # Clean up intermediate files
            try:
                os.remove(mermaid_file)
                if os.path.exists(svg_file):
                    os.remove(svg_file)
            except:
                pass

            # Return markdown image syntax
            return (
                f'\n<div class="mermaid-container">\n\n'
                f'![Architecture Diagram]({os.path.basename(png_file)})\n\n'
                f'</div>\n'
            )

        except subprocess.CalledProcessError as e:
            print(f"Error converting mermaid diagram: {e}")
            print(f"Command output: {e.stderr if e.stderr else 'No stderr'}")
            try:
                os.remove(mermaid_file)
            except:
                pass
            return f'\n```\n{mermaid_code}\n```\n'

        except Exception as e:
            print(f"Unexpected error with mermaid: {e}")
            try:
                os.remove(mermaid_file)
            except:
                pass
            return f'\n```\n{mermaid_code}\n```\n'

    return re.sub(mermaid_pattern, replace_mermaid, content, flags=re.DOTALL)

def clean_emojis_and_fix_images(content, file_dir):
    """Remove/replace emojis and fix image paths"""
    emoji_replacements = {
        '🎵': '[Audio]',
        '🎬': '[Video]',
        '📝': '[Document]',
        '📊': '[Analytics]',
        '🧠': '[AI]',
        '🎥': '[Media]',
        '📄': '[File]'
    }

    for emoji, replacement in emoji_replacements.items():
        content = content.replace(emoji, replacement)

    # Pattern to match markdown images
    img_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'

    def replace_image(match):
        alt_text = match.group(1)
        img_path = match.group(2)

        if not img_path.startswith(('http://', 'https://', '/')):
            abs_img_path = os.path.join(file_dir, img_path)
            if os.path.exists(abs_img_path):
                img_path = os.path.relpath(abs_img_path, file_dir)

        return (
            f'<img src="{img_path}" alt="{alt_text}" '
            f'style="max-width: 100%; height: auto; display: block; margin: 1em auto;" />'
        )

    content = re.sub(img_pattern, replace_image, content)

    # Fix existing HTML img tags
    content = re.sub(
        r'<img\s+([^>]*?)\s*/?>',
        lambda m: (
            f'<img {m.group(1)} '
            f'style="max-width: 100%; height: auto; display: block; margin: 1em auto;" />'
        ),
        content
    )

    return content

def main():
    if len(sys.argv) != 2:
        print("Usage: python preprocess_markdown.py <markdown_file>")
        sys.exit(1)

    md_file = sys.argv[1]

    if not os.path.exists(md_file):
        print(f"Error: File {md_file} does not exist")
        sys.exit(1)

    try:
        file_dir = os.path.dirname(os.path.abspath(md_file))

        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()

        print(f"Processing file: {md_file}")
        print(f"File directory: {file_dir}")
        print(f"Content length: {len(content)} characters")

        # Process mermaid diagrams
        content = process_mermaid_diagrams(content, file_dir)
        print(f"Mermaid processing complete. Content length: {len(content)}")

        # Clean emojis and fix image paths
        content = clean_emojis_and_fix_images(content, file_dir)
        print(f"Image path fixing complete. Content length: {len(content)}")

        # Write processed content
        processed_file = md_file.replace('.md', '_processed.md')
        with open(processed_file, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"Processed file saved as: {processed_file}")
        print(processed_file)

    except Exception as e:
        print(f"Error processing {md_file}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()