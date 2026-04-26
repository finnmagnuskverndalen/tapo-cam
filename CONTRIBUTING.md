# Contributing to Tapo C200 Camera Control

Thank you for your interest in contributing to the Tapo C200 Camera Control project! This document provides guidelines and instructions for contributing.

## 🎯 How to Contribute

### 1. Reporting Issues
- Use the GitHub issue tracker
- Search existing issues before creating new ones
- Include detailed information:
  - Steps to reproduce
  - Expected vs actual behavior
  - System information
  - Logs/output

### 2. Feature Requests
- Explain the feature clearly
- Describe the use case
- Consider if it aligns with project goals

### 3. Pull Requests
- Fork the repository
- Create a feature branch
- Make your changes
- Submit a PR with description

## 💻 Development Setup

### Prerequisites
- Rust (latest stable)
- OpenCV development libraries
- Git

### Getting Started
```bash
# Clone and setup
git clone https://github.com/yourusername/tapo-cam.git
cd tapo-cam

# Install dependencies
sudo apt-get install libopencv-dev libclang-dev clang opencv-data

# Build
cargo build --release
```

## 📝 Code Guidelines

### Rust Conventions
- Follow the official Rust style guide
- Use `cargo fmt` to format code
- Run `cargo clippy` for linting
- Add documentation comments for public APIs

### Commit Messages
- Use clear, descriptive messages
- Follow conventional commits format:
  - `feat:` New feature
  - `fix:` Bug fix
  - `docs:` Documentation
  - `style:` Formatting
  - `refactor:` Code restructuring
  - `test:` Adding tests
  - `chore:` Maintenance

### Testing
- Add tests for new functionality
- Update existing tests when modifying code
- Ensure all tests pass before submitting PR
- Test on different OpenCV versions if possible

## 🏗️ Project Structure

```
src/
├── main.rs                    # Main application entry point
├── camera.rs                  # Camera API implementation
├── detection/                 # Object detection module
└── utils/                     # Utility functions
```

## 🔍 Review Process

1. **Automated Checks**: CI must pass
2. **Code Review**: At least one maintainer approval
3. **Testing**: Verify functionality works
4. **Documentation**: Update README if needed

## 🚀 Release Process

Releases are managed by maintainers:
1. Create release branch
2. Update version in Cargo.toml
3. Update CHANGELOG.md
4. Create GitHub release
5. Tag version

## 🤝 Communication

- Be respectful and professional
- Assume good intentions
- Provide constructive feedback
- Help others when possible

## 📜 License

By contributing, you agree that your contributions will be licensed under the project's MIT License.

## 🙏 Acknowledgments

Thank you for contributing to make this project better!