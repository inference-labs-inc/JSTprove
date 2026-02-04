# Contributing to JSTprove

Thank you for your interest in contributing! Please follow these steps to ensure your contributions are smooth and
consistent with project standards.

---

## **1. Set up your development environment**

1. **Clone the repository**:

```bash
git clone https://github.com/inference-labs-inc/JSTprove.git
cd JSTprove
```

2. **Install dependencies with UV**:

```bash
uv sync --dev
uv pip install -e .
```

---

## **2. Install Git hooks**:

```bash
git config core.hooksPath hooks/
```

This points git at the tracked `hooks/` directory. The pre-commit hook automatically formats staged files (black, ruff, cargo fmt), re-stages them, then runs linters and checks (clippy, trailing whitespace, etc.).

---

## **3. Running pre-commit manually**

You can check all files in the repository at any time:

```bash
uv run pre-commit run --all-files
```

This is useful before pushing changes to catch any formatting issues early.

---

## **4. Committing changes**

1. Stage your changes:

```bash
git add <files>
```

2. Commit:

```bash
git commit -m "Your commit message"
```

The pre-commit hook will automatically format and re-stage your files. If a linter error can't be auto-fixed, the commit will be blocked and you'll need to fix it manually.

---

## **5. Pull requests**

- Always make sure your branch is up-to-date with the main branch.
- Ensure all pre-commit hooks pass locally before opening a PR.
- Run the full test suite locally and update/add tests as needed to cover your changes.
- Update documentation (README, code comments, API docs, etc.) if your changes affect usage or behavior.
- The CI pipeline will also run formatting checks and tests. Any failures must be resolved before merging.

---

## **6. Formatting & newline policy**

- **Rust files**: All `.rs` files should be formatted using `cargo fmt`.
- **Python files**: All `.py` files must have a trailing newline at EOF.
- Pre-commit hooks enforce this automatically locally and in CI.
