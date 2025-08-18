#!/bin/bash
set -e

echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
  texlive-full \
  texlive-xetex \
  texlive-luatex \
  pandoc \
  librsvg2-bin \
  python3-pip \
  nodejs \
  npm \
  imagemagick \
  ghostscript \
  wkhtmltopdf

echo "Installing Node.js dependencies for Mermaid..."
npm install -g @mermaid-js/mermaid-cli@latest
npm install -g puppeteer
sudo apt-get install -y google-chrome-stable

echo "Installing Python dependencies..."
pip install --upgrade pip
pip install \
  weasyprint \
  markdown \
  pymdown-extensions \
  pillow \
  cairosvg \
  pdfkit \
  google-auth \
  google-auth-oauthlib \
  google-auth-httplib2 \
  google-api-python-client

echo "System setup complete!"