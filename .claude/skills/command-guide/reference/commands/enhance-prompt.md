---
name: enhance-prompt
description: Enhanced prompt transformation using session memory and intent analysis with --enhance flag detection
argument-hint: "user input to enhance"
---

## Overview

Systematically enhances user prompts by leveraging session memory context and intent analysis, translating ambiguous requests into actionable specifications.

## Core Protocol

**Enhancement Pipeline:**
`Intent Translation` → `Context Integration` → `Structured Output`

**Context Sources:**
- Session memory (conversation history, previous analysis)
- Implicit technical requirements
- User intent patterns

## Enhancement Rules

### Intent Translation

| User Says | Translate To | Focus |
|-----------|--------------|-------|
| "fix" | Debug and resolve | Root cause → preserve behavior |
| "improve" | Enhance/optimize | Performance/readability |
| "add" | Implement feature | Integration + edge cases |
| "refactor" | Restructure quality | Maintain behavior |
| "update" | Modernize | Version compatibility |

### Context Integration Strategy

**Session Memory:**
- Reference recent conversation context
- Reuse previously identified patterns
- Build on established understanding
- Infer technical requirements from discussion

**Example:**
```bash
# User: "add login"
# Session Memory: Previous auth discussion, JWT mentioned
# Inferred: JWT-based auth, integrate with existing session management
# Action: Implement JWT authentication with session persistence
```

## Output Structure

```bash
INTENT: [Clear technical goal]
CONTEXT: [Session memory + codebase patterns]
ACTION: [Specific implementation steps]
ATTENTION: [Critical constraints]
```

### Output Examples

**Example 1:**
```bash
# Input: "fix login button"
INTENT: Debug non-functional login button
CONTEXT: From session - OAuth flow discussed, known state issue
ACTION: Check event binding → verify state updates → test auth flow
ATTENTION: Preserve existing OAuth integration
```

**Example 2:**
```bash
# Input: "refactor payment code"
INTENT: Restructure payment module for maintainability
CONTEXT: Session memory - PCI compliance requirements, Stripe integration patterns
ACTION: Extract reusable validators → isolate payment gateway logic → maintain adapter pattern
ATTENTION: Zero behavior change, maintain PCI compliance, full test coverage
```

## Enhancement Triggers

- Ambiguous language: "fix", "improve", "clean up"
- Vague requests requiring clarification
- Complex technical requirements
- Architecture changes
- Critical systems: auth, payment, security
- Multi-step refactoring

## Key Principles

1. **Session Memory First**: Leverage conversation context and established understanding
2. **Context Reuse**: Build on previous discussions and decisions
3. **Clear Output**: Structured, actionable specifications
4. **Intent Clarification**: Transform vague requests into specific technical goals
5. **Avoid Duplication**: Reference existing context, don't repeat