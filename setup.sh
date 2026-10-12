#!/bin/bash

# Setup script for Transformer Web App
# Usage: bash setup.sh

echo "🚀 Setting up Transformer Web App..."

# Check Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "✅ Python version: $python_version"

# Install dependencies
echo "📦 Installing dependencies..."
pip install -q -r requirements_web.txt

# Check if Tesseract is installed (optional)
echo ""
echo "🔤 Checking Tesseract OCR..."
if command -v tesseract &> /dev/null; then
    echo "✅ Tesseract is installed"
    tesseract --version | head -1
else
    echo "⚠️  Tesseract not found. OCR will use placeholder."
    echo "   To install Tesseract:"
    echo "   - Ubuntu/Debian: sudo apt-get install tesseract-ocr"
    echo "   - macOS: brew install tesseract"
    echo "   - Windows: https://github.com/UB-Mannheim/tesseract/wiki"
fi

# Download models (optional)
echo ""
echo "🤖 Pre-downloading Transformer models..."
echo "   (This may take a few minutes...)"

python -c "
import sys
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    print('   📥 Downloading MarianMT (EN→ES)...')
    AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-es')
    print('   ✅ MarianMT (EN→ES) ready')
except Exception as e:
    print(f'   ⚠️  Could not pre-download: {e}')
    print('   (Models will download on first use)')
" 2>&1 | grep -v "^  "

# Verify installation
echo ""
echo "✔️ Verifying installation..."
python -c "
import streamlit
import transformers
import torch
print('   ✅ Streamlit:', streamlit.__version__)
print('   ✅ Transformers:', transformers.__version__)
print('   ✅ PyTorch:', torch.__version__)
print('   ✅ Device: GPU' if torch.cuda.is_available() else '   ✅ Device: CPU')
"

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📝 To start the app:"
echo "   streamlit run app.py"
echo ""
echo "🌐 The app will open at:"
echo "   http://localhost:8501"
echo ""
