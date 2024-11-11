# Team8 - C1: Introduction to Human and Computer Vision

Welcome to the repository for **Team8**'s project in **C1: Introduction to Human and Computer Vision**. This repository contains all the code, data, and results for the various weekly tasks as part of the course.

Slides of the final presentation : https://docs.google.com/presentation/d/1AjJbfE0tqhlhi2BBxQSKj4C4YKtB8evcadvAOSOPUv8/edit?usp=sharing

## Project Setup

### Requirements
- **Operating System**: Windows
- **Programming Language**: Python 3.12

### Installation Guide

To set up the project on your local machine, follow these steps:

1. **Install Python**  
   We recommend using [`pyenv`](https://github.com/pyenv-win/pyenv-win) to manage your Python installations on Windows. To install Python 3.12, run the following commands in your terminal:
   ```bash
   pyenv install 3.12.x
   pyenv global 3.12.x
   ```

2. **Create Virtual Environment**  
   Once Python is installed, create a virtual environment:
   ```bash
   python -m venv venv
   ```

3. **Activate Virtual Environment**
   - For **cmd** or **PowerShell**:
     ```bash
     venv\Scripts\activate
     ```
   - For **Git Bash**:
     ```bash
     source venv/Scripts/activate
     ```

4. **Install Dependencies**  
   With the virtual environment activated, install the project dependencies from the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

The repository follows this structure:

```
/src       : Contains all the Python files to centralize common code between different weeks, such as data reading or path management.
/data      : Contains the datasets required for each week's tasks.
/WEEK_X    : Contains the code and results for each week's tasks, where X is the number of the week.
```

## Weekly Progress

Each week has a dedicated folder that contains the task-specific code and output. Ensure that the weekly data is correctly placed in the `/data` directory before running the scripts for each week.

---
