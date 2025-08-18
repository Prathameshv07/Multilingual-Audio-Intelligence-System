#!/bin/bash
set -e

SCRIPTS_DIR="$GITHUB_WORKSPACE/.github/workflows/scripts"

echo "Converting MD files to PDF..."
find . -name "*.md" -not -path "./.git/*" | while read file; do
  dir="$(dirname "$file")"
  filename="$(basename "$file" .md)"
  pdf_path="$dir/$filename.pdf"
  
  echo "Processing $file..."
  
  if [ ! -f "$file" ]; then
    echo "ERROR: File $file does not exist"
    continue
  fi
  
  if [ ! -r "$file" ]; then
    echo "ERROR: File $file is not readable"
    continue
  fi
  
  echo "File size: $(wc -c < "$file") bytes"
  
  # Preprocess the markdown file
  cd "$dir"
  processed_file=$(python3 "$SCRIPTS_DIR/preprocess_markdown.py" "$(basename "$file")" 2>&1) || {
    echo "Preprocessing failed, using original file"
    processed_file="$(basename "$file")"
  }
  
  if [ ! -f "$processed_file" ]; then
    echo "Processed file $processed_file does not exist, using original"
    processed_file="$(basename "$file")"
  fi
  
  echo "Using file for conversion: $processed_file"
  
  # Method 1: Try XeLaTeX with enhanced settings
  pandoc "$processed_file" \
    -o "$pdf_path" \
    --pdf-engine=xelatex \
    --include-in-header="$SCRIPTS_DIR/latex-header.tex" \
    --variable mainfont="DejaVu Sans" \
    --variable sansfont="DejaVu Sans" \
    --variable monofont="DejaVu Sans Mono" \
    --variable geometry:top=0.5in,left=0.5in,right=0.5in,bottom=0.5in \
    --variable colorlinks=true \
    --variable linkcolor=blue \
    --variable urlcolor=blue \
    --variable toccolor=gray \
    --resource-path="$dir:$SCRIPTS_DIR" \
    --standalone \
    --toc \
    --number-sections \
    --highlight-style=pygments \
    --wrap=auto \
    --dpi=300 \
    --verbose 2>&1 || {
    
    echo "XeLaTeX failed, trying HTML->PDF conversion..."
    
    # Method 2: HTML to PDF conversion
    pandoc "$processed_file" \
      -t html5 \
      --standalone \
      --embed-resources \
      --css="$SCRIPTS_DIR/styles.css" \
      --toc \
      --number-sections \
      --highlight-style=pygments \
      -o "$dir/$filename.html" 2>&1
    
    if [ -f "$dir/$filename.html" ]; then
      weasyprint "$dir/$filename.html" "$pdf_path" --presentational-hints 2>&1 || {
        wkhtmltopdf \
          --page-size A4 \
          --margin-top 0.5in \
          --margin-right 0.5in \
          --margin-bottom 0.5in \
          --margin-left 0.5in \
          --encoding UTF-8 \
          --no-outline \
          --enable-local-file-access \
          "$dir/$filename.html" "$pdf_path" 2>&1 || {
          echo "All conversion methods failed for $file"
          continue
        }
      }
      rm -f "$dir/$filename.html"
    else
      echo "Failed to create HTML file for $file"
      continue
    fi
  }
  
  # Clean up
  if [ "$processed_file" != "$(basename "$file")" ]; then
    rm -f "$processed_file"
  fi
  rm -f mermaid_*.png mermaid_*.svg mermaid_*.mmd
  
  if [ -f "$pdf_path" ]; then
    echo "✅ Successfully converted $file to $pdf_path"
    echo "PDF file size: $(wc -c < "$pdf_path") bytes"
  else
    echo "❌ Failed to convert $file"
  fi
  
  cd "$GITHUB_WORKSPACE"
done