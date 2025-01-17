# Azure OpenAI Chatbot

A simple chatbot implementation using Azure OpenAI and Gradio interface.

## Prerequisites Installation

### 1. Visual Studio Code
1. Download Visual Studio Code from [https://code.visualstudio.com/](https://code.visualstudio.com/)
2. Follow the installation wizard for your operating system:
   - Windows: Run the downloaded installer
   - Mac: Drag the app to Applications folder
   - Linux: Follow distribution-specific instructions
3. Install these extensions in VS Code:
    - Python
    - Pylance
    - Python Environment Manager
    - Python Indent
    - Jupyter

### 2. Python Installation
1. Download Python from [https://www.python.org/downloads/](https://www.python.org/downloads/)
   - Choose Python 3.9 or later
   - For Windows: Check "Add Python to PATH" during installation
2. Verify installation by opening terminal/command prompt:
```bash
python --version
pip --version
```

### 3. Install the packages
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# For Windows:
venv\Scripts\activate
# For Mac/Linux:
source venv/bin/activate
```

### 4. Install Required Packages
```bash
pip install -r requirements.txt
```

### 5. Update values in config.json file
```bash
pip install -r requirements.txt
```

**Development Best Practices**
- Always use virtual environment
- Don't commit config.json file
- Keep requirements.txt updated

