#!/usr/bin/env python3
"""
Analyze all command files and generate index files for command-guide skill.
"""

import os
import re
import json
import shutil
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any

# Base paths
BASE_DIR = Path("D:/Claude_dms3/.claude")
COMMANDS_DIR = BASE_DIR / "commands"
AGENTS_DIR = BASE_DIR / "agents"
SKILL_DIR = BASE_DIR / "skills" / "command-guide"
REFERENCE_DIR = SKILL_DIR / "reference"
INDEX_DIR = SKILL_DIR / "index"

def parse_frontmatter(content: str) -> Dict[str, Any]:
    """Extract YAML frontmatter from markdown content."""
    frontmatter = {}
    if content.startswith('---'):
        lines = content.split('\n')
        in_frontmatter = False
        for i, line in enumerate(lines[1:], 1):
            if line.strip() == '---':
                break
            if ':' in line:
                key, value = line.split(':', 1)
                frontmatter[key.strip()] = value.strip().strip('"')
    return frontmatter

def categorize_command(file_path: Path) -> tuple:
    """Determine category and subcategory from file path."""
    parts = file_path.relative_to(COMMANDS_DIR).parts

    if len(parts) == 1:
        return "general", None

    category = parts[0]  # cli, memory, task, workflow
    subcategory = parts[1].replace('.md', '') if len(parts) > 2 else None

    return category, subcategory

def determine_usage_scenario(name: str, description: str, category: str) -> str:
    """Determine primary usage scenario for command."""
    name_lower = name.lower()
    desc_lower = description.lower()

    # Planning indicators
    if any(word in name_lower for word in ['plan', 'design', 'breakdown', 'brainstorm']):
        return "planning"

    # Implementation indicators
    if any(word in name_lower for word in ['implement', 'execute', 'generate', 'create', 'write']):
        return "implementation"

    # Testing indicators
    if any(word in name_lower for word in ['test', 'tdd', 'verify', 'coverage']):
        return "testing"

    # Documentation indicators
    if any(word in name_lower for word in ['docs', 'documentation', 'memory']):
        return "documentation"

    # Session management indicators
    if any(word in name_lower for word in ['session', 'resume', 'status', 'complete']):
        return "session-management"

    # Analysis indicators
    if any(word in name_lower for word in ['analyze', 'review', 'diagnosis']):
        return "analysis"

    return "general"

def determine_difficulty(name: str, description: str, category: str) -> str:
    """Determine difficulty level."""
    name_lower = name.lower()

    # Beginner commands
    beginner_keywords = ['status', 'list', 'chat', 'analyze', 'version']
    if any(word in name_lower for word in beginner_keywords):
        return "Beginner"

    # Advanced commands
    advanced_keywords = ['tdd', 'conflict', 'agent', 'auto-parallel', 'coverage', 'synthesis']
    if any(word in name_lower for word in advanced_keywords):
        return "Advanced"

    # Intermediate by default
    return "Intermediate"

def analyze_command_file(file_path: Path) -> Dict[str, Any]:
    """Analyze a single command file and extract metadata."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Parse frontmatter
    frontmatter = parse_frontmatter(content)

    # Extract data
    name = frontmatter.get('name', file_path.stem)
    description = frontmatter.get('description', '')
    argument_hint = frontmatter.get('argument-hint', '')

    # Determine categorization
    category, subcategory = categorize_command(file_path)
    usage_scenario = determine_usage_scenario(name, description, category)
    difficulty = determine_difficulty(name, description, category)

    # Build relative path
    rel_path = str(file_path.relative_to(COMMANDS_DIR)).replace('\\', '/')

    # Build full command name from frontmatter name or construct it
    # If name already contains colons (e.g., "workflow:status"), use it directly
    if ':' in name:
        command_name = f"/{name}"
    elif category == "general":
        command_name = f"/{name}"
    else:
        # For subcategorized commands, build the full path
        if subcategory:
            command_name = f"/{category}:{subcategory}:{name}"
        else:
            command_name = f"/{category}:{name}"

    return {
        "name": name,
        "command": command_name,
        "description": description,
        "arguments": argument_hint,
        "category": category,
        "subcategory": subcategory,
        "usage_scenario": usage_scenario,
        "difficulty": difficulty,
        "file_path": rel_path
    }

def build_command_relationships() -> Dict[str, Any]:
    """Build command relationship mappings."""
    relationships = {
        # Workflow planning commands
        "workflow:plan": {
            "calls_internally": [
                "workflow:session:start",
                "workflow:tools:context-gather",
                "workflow:tools:conflict-resolution",
                "workflow:tools:task-generate-agent"
            ],
            "next_steps": ["workflow:action-plan-verify", "workflow:status", "workflow:execute"],
            "alternatives": ["workflow:tdd-plan"],
            "prerequisites": []
        },
        "workflow:tdd-plan": {
            "calls_internally": [
                "workflow:session:start",
                "workflow:tools:context-gather",
                "workflow:tools:task-generate-tdd"
            ],
            "next_steps": ["workflow:tdd-verify", "workflow:status", "workflow:execute"],
            "alternatives": ["workflow:plan"],
            "prerequisites": []
        },

        # Execution commands
        "workflow:execute": {
            "prerequisites": ["workflow:plan", "workflow:tdd-plan"],
            "related": ["workflow:status", "workflow:resume"],
            "next_steps": ["workflow:review", "workflow:tdd-verify"]
        },

        # Verification commands
        "workflow:action-plan-verify": {
            "prerequisites": ["workflow:plan"],
            "next_steps": ["workflow:execute"],
            "related": ["workflow:status"]
        },
        "workflow:tdd-verify": {
            "prerequisites": ["workflow:execute"],
            "related": ["workflow:tools:tdd-coverage-analysis"]
        },

        # Session management
        "workflow:session:start": {
            "next_steps": ["workflow:plan", "workflow:execute"],
            "related": ["workflow:session:list", "workflow:session:resume"]
        },
        "workflow:session:resume": {
            "alternatives": ["workflow:resume"],
            "related": ["workflow:session:list", "workflow:status"]
        },
        "workflow:resume": {
            "alternatives": ["workflow:session:resume"],
            "related": ["workflow:status"]
        },

        # Task management
        "task:create": {
            "next_steps": ["task:execute"],
            "related": ["task:breakdown"]
        },
        "task:breakdown": {
            "next_steps": ["task:execute"],
            "related": ["task:create"]
        },
        "task:replan": {
            "prerequisites": ["workflow:plan"],
            "related": ["workflow:action-plan-verify"]
        },
        "task:execute": {
            "prerequisites": ["task:create", "task:breakdown", "workflow:plan"],
            "related": ["workflow:status"]
        },

        # Memory/Documentation
        "memory:docs": {
            "calls_internally": [
                "workflow:session:start",
                "workflow:tools:context-gather"
            ],
            "next_steps": ["workflow:execute"]
        },
        "memory:skill-memory": {
            "next_steps": ["workflow:plan", "cli:analyze"],
            "related": ["memory:load-skill-memory"]
        },
        "memory:workflow-skill-memory": {
            "related": ["memory:skill-memory"],
            "next_steps": ["workflow:plan"]
        },

        # CLI modes
        "cli:execute": {
            "alternatives": ["cli:codex-execute"],
            "related": ["cli:analyze", "cli:chat"]
        },
        "cli:analyze": {
            "related": ["cli:chat", "cli:mode:code-analysis"],
            "next_steps": ["cli:execute"]
        },

        # Brainstorming
        "workflow:brainstorm:artifacts": {
            "next_steps": ["workflow:brainstorm:synthesis", "workflow:plan"],
            "related": ["workflow:brainstorm:auto-parallel"]
        },
        "workflow:brainstorm:synthesis": {
            "prerequisites": ["workflow:brainstorm:artifacts"],
            "next_steps": ["workflow:plan"]
        },
        "workflow:brainstorm:auto-parallel": {
            "next_steps": ["workflow:brainstorm:synthesis", "workflow:plan"],
            "related": ["workflow:brainstorm:artifacts"]
        },

        # Test workflows
        "workflow:test-gen": {
            "prerequisites": ["workflow:execute"],
            "next_steps": ["workflow:test-cycle-execute"]
        },
        "workflow:test-fix-gen": {
            "alternatives": ["workflow:test-gen"],
            "next_steps": ["workflow:test-cycle-execute"]
        },
        "workflow:test-cycle-execute": {
            "prerequisites": ["workflow:test-gen", "workflow:test-fix-gen"],
            "related": ["workflow:tdd-verify"]
        },

        # UI Design workflows
        "workflow:ui-design:explore-auto": {
            "calls_internally": ["workflow:ui-design:capture", "workflow:ui-design:style-extract", "workflow:ui-design:layout-extract"],
            "next_steps": ["workflow:ui-design:generate"]
        },
        "workflow:ui-design:imitate-auto": {
            "calls_internally": ["workflow:ui-design:capture"],
            "next_steps": ["workflow:ui-design:generate"]
        },

        # Lite workflows
        "workflow:lite-plan": {
            "calls_internally": ["workflow:lite-execute"],
            "next_steps": ["workflow:lite-execute", "workflow:status"],
            "alternatives": ["workflow:plan"],
            "prerequisites": []
        },
        "workflow:lite-fix": {
            "next_steps": ["workflow:lite-execute", "workflow:status"],
            "alternatives": ["workflow:lite-plan"],
            "related": ["workflow:test-cycle-execute"]
        },
        "workflow:lite-execute": {
            "prerequisites": ["workflow:lite-plan", "workflow:lite-fix"],
            "related": ["workflow:execute", "workflow:status"]
        },

        # Review cycle workflows
        "workflow:review-module-cycle": {
            "next_steps": ["workflow:review-fix"],
            "related": ["workflow:review-session-cycle", "workflow:review"]
        },
        "workflow:review-session-cycle": {
            "prerequisites": ["workflow:execute"],
            "next_steps": ["workflow:review-fix"],
            "related": ["workflow:review-module-cycle", "workflow:review"]
        },
        "workflow:review-fix": {
            "prerequisites": ["workflow:review-module-cycle", "workflow:review-session-cycle"],
            "related": ["workflow:test-cycle-execute"]
        }
    }

    return relationships

def sync_reference_directory():
    """Sync reference directory with source directories."""
    print("\n=== Syncing Reference Directory ===")

    # Step 1: Delete all files in reference directory
    if REFERENCE_DIR.exists():
        print(f"Deleting existing reference directory: {REFERENCE_DIR}")
        shutil.rmtree(REFERENCE_DIR)

    # Step 2: Create reference directory structure
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Created reference directory: {REFERENCE_DIR}")

    # Step 3: Copy agents directory
    agents_target = REFERENCE_DIR / "agents"
    if AGENTS_DIR.exists():
        print(f"Copying {AGENTS_DIR} -> {agents_target}")
        shutil.copytree(AGENTS_DIR, agents_target)
        agent_files = list(agents_target.rglob("*.md"))
        print(f"  Copied {len(agent_files)} agent files")
    else:
        print(f"  WARNING: Source directory not found: {AGENTS_DIR}")

    # Step 4: Copy commands directory
    commands_target = REFERENCE_DIR / "commands"
    if COMMANDS_DIR.exists():
        print(f"Copying {COMMANDS_DIR} -> {commands_target}")
        shutil.copytree(COMMANDS_DIR, commands_target)
        command_files = list(commands_target.rglob("*.md"))
        print(f"  Copied {len(command_files)} command files")
    else:
        print(f"  WARNING: Source directory not found: {COMMANDS_DIR}")

    print("Reference directory sync completed\n")

def identify_essential_commands(all_commands: List[Dict]) -> List[Dict]:
    """Identify the most essential commands for beginners."""
    # Essential command names (14 most important) - use full command paths
    essential_names = [
        "workflow:lite-plan",
        "workflow:lite-fix",
        "workflow:plan",
        "workflow:execute",
        "workflow:status",
        "workflow:session:start",
        "workflow:review-session-cycle",
        "cli:analyze",
        "cli:chat",
        "memory:docs",
        "workflow:brainstorm:artifacts",
        "workflow:action-plan-verify",
        "workflow:resume",
        "version"
    ]

    essential = []
    for cmd in all_commands:
        # Check command name without leading slash
        cmd_name = cmd['command'].lstrip('/')
        if cmd_name in essential_names:
            essential.append(cmd)

    # Sort by order in essential_names
    essential.sort(key=lambda x: essential_names.index(x['command'].lstrip('/')))

    return essential[:14]  # Limit to 14

def main():
    """Main analysis function."""
    import sys
    import io

    # Fix Windows console encoding
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("=== Command Guide Index Rebuild ===\n")

    # Step 1: Sync reference directory
    sync_reference_directory()

    # Step 2: Analyze command files
    print("=== Analyzing Command Files ===")

    # Find all command files
    command_files = list(COMMANDS_DIR.rglob("*.md"))
    print(f"Found {len(command_files)} command files")

    # Analyze each command
    all_commands = []
    for cmd_file in sorted(command_files):
        try:
            metadata = analyze_command_file(cmd_file)
            all_commands.append(metadata)
            print(f"  OK {metadata['command']}")
        except Exception as e:
            print(f"  ERROR analyzing {cmd_file}: {e}")

    print(f"\nAnalyzed {len(all_commands)} commands")

    # Generate index files
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # 1. all-commands.json
    all_commands_path = INDEX_DIR / "all-commands.json"
    with open(all_commands_path, 'w', encoding='utf-8') as f:
        json.dump(all_commands, f, indent=2, ensure_ascii=False)
    print(f"\nOK Generated {all_commands_path} ({os.path.getsize(all_commands_path)} bytes)")

    # 2. by-category.json
    by_category = defaultdict(lambda: defaultdict(list))
    for cmd in all_commands:
        cat = cmd['category']
        subcat = cmd['subcategory'] or '_root'
        by_category[cat][subcat].append(cmd)

    by_category_path = INDEX_DIR / "by-category.json"
    with open(by_category_path, 'w', encoding='utf-8') as f:
        json.dump(dict(by_category), f, indent=2, ensure_ascii=False)
    print(f"OK Generated {by_category_path} ({os.path.getsize(by_category_path)} bytes)")

    # 3. by-use-case.json
    by_use_case = defaultdict(list)
    for cmd in all_commands:
        by_use_case[cmd['usage_scenario']].append(cmd)

    by_use_case_path = INDEX_DIR / "by-use-case.json"
    with open(by_use_case_path, 'w', encoding='utf-8') as f:
        json.dump(dict(by_use_case), f, indent=2, ensure_ascii=False)
    print(f"OK Generated {by_use_case_path} ({os.path.getsize(by_use_case_path)} bytes)")

    # 4. essential-commands.json
    essential = identify_essential_commands(all_commands)
    essential_path = INDEX_DIR / "essential-commands.json"
    with open(essential_path, 'w', encoding='utf-8') as f:
        json.dump(essential, f, indent=2, ensure_ascii=False)
    print(f"OK Generated {essential_path} ({os.path.getsize(essential_path)} bytes)")

    # 5. command-relationships.json
    relationships = build_command_relationships()
    relationships_path = INDEX_DIR / "command-relationships.json"
    with open(relationships_path, 'w', encoding='utf-8') as f:
        json.dump(relationships, f, indent=2, ensure_ascii=False)
    print(f"OK Generated {relationships_path} ({os.path.getsize(relationships_path)} bytes)")

    # Print summary statistics
    print("\n=== Summary Statistics ===")

    # Reference directory statistics
    if REFERENCE_DIR.exists():
        ref_agents = list((REFERENCE_DIR / "agents").rglob("*.md")) if (REFERENCE_DIR / "agents").exists() else []
        ref_commands = list((REFERENCE_DIR / "commands").rglob("*.md")) if (REFERENCE_DIR / "commands").exists() else []
        print(f"\nReference directory:")
        print(f"  Agents: {len(ref_agents)} files")
        print(f"  Commands: {len(ref_commands)} files")
        print(f"  Total: {len(ref_agents) + len(ref_commands)} files")

    print(f"\nTotal commands indexed: {len(all_commands)}")
    print(f"\nBy category:")
    for cat in sorted(by_category.keys()):
        total = sum(len(cmds) for cmds in by_category[cat].values())
        print(f"  {cat}: {total}")
        for subcat in sorted(by_category[cat].keys()):
            if subcat != '_root':
                print(f"    - {subcat}: {len(by_category[cat][subcat])}")

    print(f"\nBy usage scenario:")
    for scenario in sorted(by_use_case.keys()):
        print(f"  {scenario}: {len(by_use_case[scenario])}")

    print(f"\nBy difficulty:")
    difficulty_counts = defaultdict(int)
    for cmd in all_commands:
        difficulty_counts[cmd['difficulty']] += 1
    for difficulty in ['Beginner', 'Intermediate', 'Advanced']:
        print(f"  {difficulty}: {difficulty_counts[difficulty]}")

    print(f"\nEssential commands: {len(essential)}")

    print("\n=== Index Rebuild Complete ===")
    print(f"Reference: {REFERENCE_DIR}")
    print(f"Index: {INDEX_DIR}")

if __name__ == '__main__':
    main()
