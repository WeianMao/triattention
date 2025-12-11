---
name: javascript-dev
description: JavaScript and Node.js core development principles and essential practices
---

# JavaScript Development Guidelines

You are now operating under JavaScript/Node.js core development principles. Focus on essential practices without dictating project structure.

## Core Language Principles

### Naming Conventions
- **Variables/Functions**: camelCase (`getUserData`, `isValid`)
- **Constants**: SCREAMING_SNAKE_CASE (`API_ENDPOINT`, `MAX_RETRIES`)
- **Classes**: PascalCase (`UserService`, `ApiClient`)

### Essential Function Guidelines
- **Pure Functions**: Prefer functions that don't mutate inputs
- **Async/Await**: Use instead of Promises for better readability
- **Error Handling**: Always handle errors explicitly, never silently fail

```javascript
// Core principle: Clear error handling
async function fetchData(id) {
  try {
    const response = await api.get(`/data/${id}`);
    return response.data;
  } catch (error) {
    throw new DataFetchError(`Failed to fetch data for ${id}: ${error.message}`);
  }
}
```

## Essential Testing Practices
- **Test Names**: Describe behavior clearly (`should return user when ID exists`)
- **Arrange-Act-Assert**: Structure tests consistently
- **Mock External Dependencies**: Isolate units under test
- **Test Edge Cases**: Include error conditions and boundary values

## Code Quality Essentials
- **Consistent Formatting**: Use automated formatting (Prettier)
- **Linting**: Catch common errors early (ESLint)
- **Type Safety**: Consider TypeScript for larger projects
- **Input Validation**: Validate all external inputs

## Security Core Principles
- **Input Sanitization**: Never trust user input
- **Environment Variables**: Keep secrets out of code
- **Dependency Management**: Regularly audit and update packages
- **Error Messages**: Don't expose internal details to users

## Performance Guidelines
- **Avoid Premature Optimization**: Write clear code first
- **Use Modern Array Methods**: `map`, `filter`, `reduce` over manual loops
- **Template Literals**: For string formatting over concatenation
- **Object Destructuring**: For cleaner variable extraction

Apply these core JavaScript principles to ensure clean, maintainable, and secure code without imposing specific project structures or tool choices.