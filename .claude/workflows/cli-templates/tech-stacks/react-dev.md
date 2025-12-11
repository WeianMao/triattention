---
name: react-dev
description: React core development principles with hooks and modern patterns
---

# React Development Guidelines

You are now operating under React core development principles. Focus on essential React patterns without dictating project structure.

## Core React Principles

### Component Guidelines
- **Functional Components**: Use functional components with hooks over class components
- **Component Naming**: PascalCase for components (`UserProfile`, `NavigationBar`)
- **Single Responsibility**: Each component should have one clear purpose
- **Props Interface**: Define clear prop types (TypeScript when possible)

### Essential Hook Patterns
- **useState**: For component-level state management
- **useEffect**: Handle side effects with proper cleanup
- **useCallback**: Memoize functions to prevent unnecessary re-renders
- **useMemo**: Memoize expensive calculations
- **Custom Hooks**: Extract reusable stateful logic

```jsx
// Core principle: Custom hooks for reusable logic
function useUser(userId) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!userId) return;
    
    const fetchUser = async () => {
      try {
        setLoading(true);
        const userData = await api.getUser(userId);
        setUser(userData);
      } catch (err) {
        setError(err);
      } finally {
        setLoading(false);
      }
    };

    fetchUser();
  }, [userId]);

  return { user, loading, error };
}
```

## Essential Testing Practices
- **Test Behavior**: Test what users see and interact with, not implementation
- **React Testing Library**: Preferred over enzyme for component testing
- **Mock External Dependencies**: Keep tests isolated and fast
- **Accessibility Testing**: Include accessibility assertions in tests

## Performance Core Principles
- **Avoid Premature Optimization**: Write clear code first
- **React.memo**: Only when profiling shows unnecessary re-renders
- **useCallback/useMemo**: Use judiciously based on actual performance needs
- **Key Props**: Always provide stable, unique keys for list items

## State Management Guidelines
- **Local State First**: Use useState for component-specific state
- **Lift State Up**: When multiple components need the same data
- **Context Sparingly**: Only for truly global state (theme, auth, etc.)
- **External Libraries**: Consider Redux Toolkit, Zustand for complex global state

```jsx
// Core principle: Context with proper error boundaries
const UserContext = createContext();

export const useUser = () => {
  const context = useContext(UserContext);
  if (!context) {
    throw new Error('useUser must be used within UserProvider');
  }
  return context;
};
```

## Error Handling Essentials
- **Error Boundaries**: Implement to catch React component errors
- **Async Error Handling**: Use try-catch in useEffect and event handlers
- **User-Friendly Messages**: Show helpful error states to users
- **Error Recovery**: Provide ways for users to recover from errors

## Accessibility Core Principles
- **Semantic HTML**: Use appropriate HTML elements first
- **ARIA Labels**: Add when semantic HTML isn't sufficient
- **Keyboard Navigation**: Ensure all interactive elements are keyboard accessible
- **Focus Management**: Handle focus properly for dynamic content

## Code Quality Essentials
- **ESLint React Rules**: Use React-specific linting rules
- **TypeScript**: Use for prop types and state management
- **Consistent Formatting**: Use Prettier or similar
- **Component Composition**: Favor composition over complex inheritance

Apply these core React principles to ensure maintainable, performant, and accessible components without imposing specific architectural decisions.