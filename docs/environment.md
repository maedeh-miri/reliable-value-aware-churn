# Reproducible Python Environment

## Reference environment

The environment used for the initial data audits and split validation was:

- Operating system: Windows
- Python: 3.13.2
- pandas: 3.0.6
- scikit-learn: 1.9.1

## Dependency files

`requirements.txt` records the direct third-party dependencies identified in the current experiment scripts.

`requirements-lock.txt` records the complete output of `pip freeze` from the reference virtual environment, including transitive and potentially incidental packages.

## Create a virtual environment

From the repository root in PowerShell:

```powershell
py -3.13 -m venv .venv
```

This command requires Python 3.13 to be installed and available through the Windows Python launcher.

## Install dependencies

For the project's declared direct dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

To attempt to recreate the recorded package versions:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

## Verify the environment

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pip freeze
```

`pip check` verifies whether installed packages have compatible declared dependencies. It does not test the project's application logic.

## Reproducibility limitations

The lock file is a snapshot of the Windows reference environment. It does not record distribution-file hashes or guarantee identical installation on other operating systems.

The original raw dataset is not tracked in Git. Dataset acquisition and SHA-256 verification are separate requirements for reproducing data-dependent experiments.

The reference environment has been inventoried, but a clean-room installation from these files has not yet been verified.
