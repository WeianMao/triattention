# Tool Strategy

## ⚡ Exa Triggering Mechanisms

**Auto-Trigger**:
- User mentions "exa-code" or code-related queries → `mcp__exa__get_code_context_exa`
- Need current web information → `mcp__exa__web_search_exa`

**Manual Trigger**:
- Complex API research → Exa Code Context
- Real-time information needs → Exa Web Search

## ⚡ CCW edit_file Tool (AI-Powered Editing)

**When to Use**: Edit tool fails 1+ times on same file

### Usage

**Best for**: Code block replacements, function rewrites, multi-line changes

```bash
ccw tool exec edit_file --path "file.py" --old "def old():
    pass" --new "def new():
    return True"
```

**Parameters**:
- `--path`: File path to edit
- `--old`: Text to find and replace
- `--new`: New text to insert

**Features**:
- ✅ Exact text matching (precise and predictable)
- ✅ Auto line ending adaptation (CRLF/LF)
- ✅ No JSON escaping issues
- ✅ Multi-line text supported with quotes

### Fallback Strategy

1. **Edit fails 1+ times** → Use `ccw tool exec edit_file`
2. **Still fails** → Use Write to recreate file

## ⚡ sed Line Operations (Line Mode Alternative)

**When to Use**: Precise line number control (insert, delete, replace specific lines)

### Common Operations

```bash
# Insert after line 10
sed -i '10a\new line content' file.txt

# Insert before line 5
sed -i '5i\new line content' file.txt

# Delete line 3
sed -i '3d' file.txt

# Delete lines 5-8
sed -i '5,8d' file.txt

# Replace line 3 content
sed -i '3c\replacement line' file.txt

# Replace lines 3-5 content
sed -i '3,5c\single replacement line' file.txt
```

### Operation Reference

| Operation | Command | Example |
|-----------|---------|---------|
| Insert after | `Na\text` | `sed -i '10a\new' file` |
| Insert before | `Ni\text` | `sed -i '5i\new' file` |
| Delete line | `Nd` | `sed -i '3d' file` |
| Delete range | `N,Md` | `sed -i '5,8d' file` |
| Replace line | `Nc\text` | `sed -i '3c\new' file` |

**Note**: Use `sed -i` for in-place file modification (works in Git Bash on Windows)
