---
name: java-dev
description: Java core development principles for robust, maintainable enterprise applications
---

# Java Development Guidelines

You are now operating under Java core development principles. Focus on essential Java practices without dictating specific frameworks or project structure.

## Core Java Principles

### Essential Language Guidelines
- **Object-Oriented Design**: Use proper encapsulation, inheritance, and polymorphism
- **Naming Conventions**: Follow Java naming standards (camelCase, PascalCase)
- **Immutability**: Favor immutable objects when possible
- **Exception Handling**: Use specific exceptions and proper exception handling

```java
// Core principle: Proper exception handling and immutability
public final class User {
    private final String id;
    private final String name;
    private final String email;

    public User(String id, String name, String email) {
        this.id = Objects.requireNonNull(id, "User ID cannot be null");
        this.name = Objects.requireNonNull(name, "Name cannot be null");
        this.email = Objects.requireNonNull(email, "Email cannot be null");
    }

    public String getId() { return id; }
    public String getName() { return name; }
    public String getEmail() { return email; }
}

// Core principle: Specific exceptions
public class UserNotFoundException extends Exception {
    public UserNotFoundException(String message) {
        super(message);
    }
}
```

## Essential Object-Oriented Patterns
- **Single Responsibility**: Each class should have one reason to change
- **Dependency Injection**: Use constructor injection for required dependencies
- **Interface Segregation**: Keep interfaces focused and minimal
- **Composition over Inheritance**: Favor composition for flexibility

## Error Handling Essentials
- **Checked vs Unchecked**: Use checked exceptions for recoverable conditions
- **Exception Hierarchy**: Create meaningful exception hierarchies
- **Resource Management**: Use try-with-resources for automatic resource cleanup
- **Logging**: Log exceptions appropriately with context

```java
// Core principle: Resource management and exception handling
public class UserService {
    private final UserRepository userRepository;
    
    public UserService(UserRepository userRepository) {
        this.userRepository = Objects.requireNonNull(userRepository);
    }
    
    public User findUser(String id) throws UserNotFoundException {
        try {
            Optional<User> user = userRepository.findById(id);
            return user.orElseThrow(() -> 
                new UserNotFoundException("User not found with id: " + id));
        } catch (DataAccessException e) {
            throw new UserNotFoundException("Error retrieving user: " + e.getMessage(), e);
        }
    }
}
```

## Modern Java Features
- **Optional**: Use Optional to handle null values safely
- **Streams**: Use streams for data processing and filtering
- **Lambda Expressions**: Use lambdas for functional programming patterns
- **Records**: Use records for simple data carriers (Java 14+)

## Testing Essentials
- **Unit Tests**: Use JUnit 5 for unit testing
- **Test Organization**: Follow Given-When-Then or Arrange-Act-Assert patterns
- **Mocking**: Use Mockito for mocking dependencies
- **Integration Tests**: Test component interactions properly

## Performance Guidelines
- **String Handling**: Use StringBuilder for string concatenation in loops
- **Collection Choice**: Choose appropriate collection types for use cases
- **Memory Management**: Understand garbage collection behavior
- **Profiling**: Use profiling tools to identify performance bottlenecks

## Concurrency Guidelines
- **Thread Safety**: Design for thread safety when needed
- **Concurrent Collections**: Use concurrent collections appropriately
- **ExecutorService**: Use thread pools instead of creating threads manually
- **Synchronization**: Use proper synchronization mechanisms

## Code Quality Essentials
- **Code Formatting**: Use consistent code formatting and style
- **Static Analysis**: Use tools like SpotBugs, PMD, and Checkstyle
- **Documentation**: Write clear Javadoc for public APIs
- **Clean Code**: Follow clean code principles and naming conventions

Apply these core Java principles to write robust, maintainable, and efficient Java applications following language best practices and modern Java idioms.