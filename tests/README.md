# Pipe Controller Tests

This directory contains unit tests for the pipe controller functionality.

## Running Tests

### Run all tests:
```bash
cd FlapPyBird
python -m pytest tests/ -v
```

### Run specific test file:
```bash
python -m pytest tests/test_pipe_controller.py -v
python -m pytest tests/test_pipe_controller_agent.py -v
python -m pytest tests/test_ai_server_integration.py -v
```

### Run with unittest:
```bash
python -m unittest discover tests -v
```

## Test Files

- **test_pipe_controller.py**: Tests for the PipeController neural network
  - Network initialization
  - Forward pass (single and batch)
  - Output range validation
  - Gradient flow

- **test_pipe_controller_agent.py**: Tests for PipeControllerAgent
  - Agent initialization
  - Action selection
  - Experience storage
  - Training/optimization
  - Checkpoint save/load

- **test_ai_server_integration.py**: Integration tests
  - State extraction from environment
  - User override functionality
  - NN control when no users
  - State normalization
  - Experience collection format

## Requirements

Tests require:
- pytest (optional, for pytest runner)
- unittest (built-in)
- torch
- numpy

Install pytest if needed:
```bash
pip install pytest
```

