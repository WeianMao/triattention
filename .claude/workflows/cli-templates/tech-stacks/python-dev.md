---
name: python-dev
description: Python core development principles following PEP 8 and essential practices
---

# Python Development Guidelines

You are now operating under Python core development principles. Focus on essential PEP 8 practices without dictating project structure.

## Core Language Principles

### Naming Conventions (PEP 8)
- **Variables/Functions**: snake_case (`get_user_data`, `is_valid`)
- **Constants**: SCREAMING_SNAKE_CASE (`API_ENDPOINT`, `MAX_RETRIES`)
- **Classes**: PascalCase (`UserService`, `ApiClient`)
- **Private**: Single underscore prefix (`_private_method`)

### Essential Function Guidelines
- **Type Hints**: Use for parameters and return values when helpful
- **Single Responsibility**: Each function should do one thing well
- **Explicit Error Handling**: Create specific exception classes
- **Context Managers**: Use `with` statements for resource management

```python
from typing import List, Optional

def calculate_total(items: List[dict]) -> float:
    """Calculate total price of items."""
    if not items:
        raise ValueError("Items list cannot be empty")
    return sum(item.get('price', 0) for item in items)

# Core principle: Specific exceptions
class UserNotFoundError(Exception):
    """Raised when user cannot be found."""
    pass
```

## Essential Testing Practices
- **Test Structure**: Given-When-Then pattern
- **Descriptive Names**: Test names should describe behavior
- **Mock External Dependencies**: Isolate units under test
- **Edge Cases**: Test error conditions and boundary values

## Code Quality Essentials
- **PEP 8 Compliance**: Follow standard Python style guide
- **Type Checking**: Use mypy or similar for type safety
- **Automated Formatting**: Use Black or similar formatter
- **Import Organization**: Keep imports organized and minimal

## Security Core Principles
- **Input Validation**: Validate all external inputs
- **Parameterized Queries**: Never use string interpolation for SQL
- **Environment Variables**: Keep secrets out of code
- **Dependency Management**: Regularly audit packages for vulnerabilities

```python
# Core principle: Safe database queries
from sqlalchemy import text

def get_user_by_email(email: str) -> Optional[User]:
    query = text("SELECT * FROM users WHERE email = :email")
    result = db.execute(query, {"email": email})
    return result.fetchone()
```

## Modern Python Features
- **F-strings**: Use for string formatting (`f"Hello {name}"`)
- **Pathlib**: Use `pathlib.Path` instead of `os.path` 
- **Dataclasses**: Use for simple data containers
- **List/Dict Comprehensions**: Use appropriately for clarity

## Performance Guidelines
- **Avoid Premature Optimization**: Write clear code first
- **Use Built-in Functions**: Leverage Python's built-in efficiency
- **Generator Expressions**: For memory efficiency with large datasets
- **Context Managers**: Ensure proper resource cleanup

Apply these core Python principles to ensure clean, maintainable, and Pythonic code without imposing specific frameworks or project structures.