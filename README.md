# PCB Quality Assurance Toolkit

This is an internal tool to perform quality assurance on PCBs through computer vision technique based on [ChangeChip](https://github.com/Scientific-Computing-Lab/ChangeChip)

## Installation

### Prerequisites
- Tested on Python 3.11+ on Windows

### Using `venv`

1. Clone the repository:
    ```sh
    git clone https://github.com/ooferdoodles1337/pcb-qa-toolkit
    ```
2. Navigate to the project directory:
    ```sh
    cd pcb-qa-toolkit
    ```
3. Create and activate a virtual environment:
    ```sh
    python -m venv venv
    venv/Scripts/activate  # On Windows
    # source venv/bin/activate  # On macOS/Linux
    ```
4. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

### Using `conda`

1. Clone the repository:
    ```sh
    git clone https://github.com/ooferdoodles1337/pcb-qa-toolkit
    ```
2. Navigate to the project directory:
    ```sh
    cd pcb-qa-toolkit
    ```
3. Create and activate a conda environment:
    ```sh
    conda env create -n pcb python=3.11
    conda activate pcb
    ```
4. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Running the Application

To run the application, execute:
```sh
python app.py
```

