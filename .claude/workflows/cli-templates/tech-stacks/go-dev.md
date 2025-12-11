---
name: go-dev
description: Go core development principles for clean, efficient, and idiomatic code
---

# Go Development Guidelines

You are now operating under Go core development principles. Focus on essential Go idioms and practices without dictating project structure.

## Core Go Principles

### Essential Language Guidelines
- **Simplicity**: Write simple, readable code over clever solutions
- **Naming**: Use clear, descriptive names following Go conventions
- **Error Handling**: Handle errors explicitly, don't ignore them
- **Interfaces**: Keep interfaces small and focused

```go
// Core principle: Clear error handling
func GetUser(id string) (*User, error) {
    if id == "" {
        return nil, errors.New("user ID cannot be empty")
    }
    
    user, err := database.FindUser(id)
    if err != nil {
        return nil, fmt.Errorf("failed to get user %s: %w", id, err)
    }
    
    return user, nil
}

// Core principle: Small, focused interfaces
type UserReader interface {
    GetUser(id string) (*User, error)
}
```

## Idiomatic Go Patterns
- **Zero Values**: Design types to be useful in their zero state
- **Receiver Types**: Use pointer receivers for methods that modify the receiver
- **Package Names**: Use short, clear package names without underscores
- **Goroutines**: Use goroutines and channels for concurrent operations

## Essential Error Handling
- **Explicit Errors**: Always handle errors explicitly
- **Error Wrapping**: Use `fmt.Errorf` with `%w` verb to wrap errors
- **Custom Errors**: Create specific error types when appropriate
- **Early Returns**: Use early returns to avoid deep nesting

```go
// Core principle: Error wrapping and context
func ProcessUserData(userID string) error {
    user, err := GetUser(userID)
    if err != nil {
        return fmt.Errorf("processing user data: %w", err)
    }
    
    if err := validateUser(user); err != nil {
        return fmt.Errorf("user validation failed: %w", err)
    }
    
    return nil
}
```

## Concurrency Guidelines
- **Channel Communication**: Use channels to communicate between goroutines
- **Context**: Use context.Context for cancellation and timeouts
- **Worker Pools**: Implement worker pools for bounded concurrency
- **Race Detection**: Run tests with `-race` flag regularly

## Testing Essentials
- **Table-Driven Tests**: Use table-driven tests for multiple test cases
- **Test Names**: Use descriptive test function names
- **Mocking**: Use interfaces for dependency injection and mocking
- **Benchmarks**: Write benchmarks for performance-critical code

## Performance Guidelines
- **Profiling**: Use Go's built-in profiling tools
- **Memory Management**: Understand Go's garbage collector behavior
- **Slice/Map Operations**: Be aware of capacity vs length for slices
- **String Operations**: Use strings.Builder for efficient string concatenation

## Code Quality Essentials
- **Go fmt**: Always format code with `gofmt` or `goimports`
- **Go vet**: Run `go vet` to catch common mistakes
- **Linting**: Use golangci-lint for comprehensive code analysis
- **Documentation**: Write clear package and function documentation

Apply these core Go principles to write clean, efficient, and maintainable Go code following language idioms and best practices.