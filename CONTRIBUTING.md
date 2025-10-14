# Contributing to Prototype Bottleneck Model

Thank you for your interest in contributing to the Prototype Bottleneck Model! This document provides guidelines for contributing to the project.

## 🤝 How to Contribute

### 1. Fork the Repository

1. Go to the [main repository](https://github.com/yourusername/prototype-bottleneck-model)
2. Click the "Fork" button in the top right corner
3. Clone your forked repository to your local machine

### 2. Set Up Development Environment

```bash
# Clone your fork
git clone https://github.com/yourusername/prototype-bottleneck-model.git
cd prototype-bottleneck-model

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -e ".[dev]"
```

### 3. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 4. Make Your Changes

- Write clear, well-documented code
- Add tests for new functionality
- Update documentation as needed
- Follow the existing code style

### 5. Test Your Changes

```bash
# Run the test suite
python test_model.py

# Run additional tests (if you add them)
pytest tests/

# Check code style
black .
flake8 .
```

### 6. Commit Your Changes

```bash
git add .
git commit -m "Add: brief description of your changes"
```

### 7. Push and Create a Pull Request

```bash
git push origin feature/your-feature-name
```

Then go to your fork on GitHub and create a Pull Request.

## 📋 Contribution Guidelines

### Code Style

- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and concise

### Documentation

- Update README.md if you add new features
- Add docstrings to new functions and classes
- Update examples if you change the API

### Testing

- Add tests for new functionality
- Ensure all existing tests pass
- Test on different Python versions if possible

### Commit Messages

Use clear, descriptive commit messages:

- `Add: new feature description`
- `Fix: bug description`
- `Update: improvement description`
- `Docs: documentation update`
- `Test: test addition or update`

## 🎯 Areas for Contribution

### High Priority

- **Performance Improvements**: Optimize model training and inference
- **Additional Datasets**: Add examples for different data types
- **Advanced Visualizations**: Enhance the visualization tools
- **Documentation**: Improve tutorials and examples

### Medium Priority

- **New Architectures**: Extend the model with different bottleneck designs
- **Evaluation Metrics**: Add more interpretability metrics
- **Integration**: Add support for other frameworks (TensorFlow, JAX)
- **Deployment**: Add model serving capabilities

### Low Priority

- **UI/UX**: Create web interfaces for model exploration
- **Mobile**: Add mobile deployment support
- **Cloud**: Add cloud deployment examples

## 🐛 Reporting Issues

When reporting issues, please include:

1. **Environment**: Python version, OS, package versions
2. **Reproduction**: Steps to reproduce the issue
3. **Expected vs Actual**: What you expected vs what happened
4. **Code**: Minimal code example if applicable
5. **Error Messages**: Full error traceback

## 📝 Pull Request Process

1. **Description**: Provide a clear description of your changes
2. **Testing**: Ensure all tests pass
3. **Documentation**: Update relevant documentation
4. **Review**: Address any review comments
5. **Merge**: Once approved, your PR will be merged

## 🏷️ Issue Labels

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Improvements or additions to documentation
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention is needed
- `question`: Further information is requested

## 📞 Getting Help

- **Issues**: Use GitHub issues for bugs and feature requests
- **Discussions**: Use GitHub discussions for questions and ideas
- **Email**: Contact the maintainers directly for sensitive issues

## 🎉 Recognition

Contributors will be recognized in:

- The project README
- Release notes
- Contributor hall of fame (if we create one)

Thank you for contributing to making the Prototype Bottleneck Model better! 🚀
