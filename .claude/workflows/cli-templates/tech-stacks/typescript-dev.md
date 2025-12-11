---
name: typescript-dev
description: TypeScript core principles for type safety and maintainable code
---

# TypeScript Development Guidelines

You are now operating under TypeScript core principles. Focus on essential type safety practices without dictating project structure.

## Core TypeScript Principles

### Essential Type Guidelines
- **Strict Configuration**: Enable strict TypeScript settings for maximum type safety
- **Explicit Typing**: Type function parameters and return values explicitly
- **Interface vs Type**: Use interfaces for objects, types for unions and computations
- **Generic Constraints**: Use generics for reusable, type-safe components

```typescript
// Core principle: Clear type definitions
interface User {
  id: string;
  name: string;
  email: string;
  createdAt: Date;
}

// Core principle: Generic constraints
function findById<T extends { id: string }>(items: T[], id: string): T | undefined {
  return items.find(item => item.id === id);
}

// Core principle: Union types for controlled values
type Status = 'pending' | 'approved' | 'rejected';
```

## Type Safety Essentials
- **No Any**: Avoid `any` type, use `unknown` when type is truly unknown
- **Null Safety**: Handle null/undefined explicitly with strict null checks
- **Type Guards**: Use type predicates and guards for runtime type checking
- **Assertion Functions**: Create functions that narrow types safely

```typescript
// Core principle: Type guards for runtime safety
function isUser(value: unknown): value is User {
  return typeof value === 'object' && 
         value !== null && 
         typeof (value as User).id === 'string';
}

// Core principle: Never use any, use unknown
function processApiResponse(response: unknown): User | null {
  if (isUser(response)) {
    return response;
  }
  return null;
}
```

## Essential Error Handling
- **Typed Errors**: Create specific error types for different failure modes
- **Result Types**: Consider Result/Either patterns for error handling
- **Promise Typing**: Properly type async functions and error states
- **Error Boundaries**: Type error boundary props and state properly

## Performance Guidelines
- **Avoid Excessive Generics**: Don't over-engineer type parameters
- **Compile-Time Checks**: Leverage TypeScript for compile-time validation
- **Type Inference**: Let TypeScript infer types when obvious
- **Utility Types**: Use built-in utility types (Partial, Pick, Omit, etc.)

## Testing with Types
- **Test Type Assertions**: Ensure tests validate both runtime and compile-time behavior
- **Mock Typing**: Properly type test mocks and fixtures
- **Type-Only Tests**: Use TypeScript compiler API for pure type testing
- **Coverage**: Include type coverage in your quality metrics

## Configuration Essentials
- **Strict Mode**: Always enable strict mode in tsconfig.json
- **Path Mapping**: Use path mapping for cleaner imports
- **Incremental Compilation**: Enable for faster builds
- **Source Maps**: Generate source maps for debugging

Apply these core TypeScript principles to ensure type-safe, maintainable code without imposing specific architectural patterns or tooling choices.