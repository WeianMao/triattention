#!/bin/bash

##############################################################################
# 命令索引更新脚本
# 用途: 维护者使用此脚本重新生成命令索引文件
# 使用: bash update-index.sh
##############################################################################

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(dirname "$SCRIPT_DIR")"
COMMANDS_DIR="$(dirname "$(dirname "$SKILL_DIR")")/commands"
INDEX_DIR="$SKILL_DIR/index"

echo -e "${GREEN}=== 命令索引更新工具 ===${NC}"
echo ""
echo "技能目录: $SKILL_DIR"
echo "命令目录: $COMMANDS_DIR"
echo "索引目录: $INDEX_DIR"
echo ""

# 检查命令目录是否存在
if [ ! -d "$COMMANDS_DIR" ]; then
    echo -e "${RED}错误: 命令目录不存在: $COMMANDS_DIR${NC}"
    exit 1
fi

# 检查 gemini 是否可用
if ! command -v gemini &> /dev/null; then
    echo -e "${RED}错误: gemini 命令未找到${NC}"
    echo "请确保 gemini CLI 工具已安装并在 PATH 中"
    exit 1
fi

# 统计命令文件数量
COMMAND_COUNT=$(find "$COMMANDS_DIR" -name "*.md" -type f | wc -l)
echo -e "${YELLOW}发现 $COMMAND_COUNT 个命令文件${NC}"
echo ""

# 备份现有索引（如果存在）
if [ -d "$INDEX_DIR" ] && [ "$(ls -A $INDEX_DIR)" ]; then
    BACKUP_DIR="$INDEX_DIR/.backup-$(date +%Y%m%d-%H%M%S)"
    echo -e "${YELLOW}备份现有索引到: $BACKUP_DIR${NC}"
    mkdir -p "$BACKUP_DIR"
    cp "$INDEX_DIR"/*.json "$BACKUP_DIR/" 2>/dev/null || true
    echo ""
fi

# 确保索引目录存在
mkdir -p "$INDEX_DIR"

echo -e "${GREEN}开始生成索引...${NC}"
echo ""

# 使用 gemini 生成索引
cd "$COMMANDS_DIR" && gemini -p "
PURPOSE: 解析所有命令文件（约 $COMMAND_COUNT 个）并重新生成结构化命令索引
TASK:
• 扫描所有 .md 命令文件（包括子目录）
• 提取每个命令的元数据：name, description, arguments, category, subcategory
• 分析命令的使用场景和难度等级（初级/中级/高级）
• 识别命令之间的关联关系（通常一起使用的命令、前置依赖、后续推荐）
• 识别10-15个最常用的核心命令
• 按使用场景分类（planning/implementation/testing/documentation/session-management）
• 生成5个 JSON 索引文件到 $INDEX_DIR 目录
MODE: write
CONTEXT: @**/*.md | Memory: 命令分为4大类：workflow, cli, memory, task，需要理解每个命令的实际用途来生成准确的索引
EXPECTED: 生成5个规范的 JSON 文件：
1. all-commands.json - 包含所有命令的完整信息数组
2. by-category.json - 按 category/subcategory 层级组织
3. by-use-case.json - 按使用场景（planning/implementation/testing等）组织
4. essential-commands.json - 核心命令列表（10-15个）
5. command-relationships.json - 命令关联关系图（command -> related_commands数组）
每个命令对象包含：name, description, arguments, category, subcategory, usage_scenario, difficulty, file_path
RULES: 保持一致的数据结构，JSON格式严格遵循规范，确保所有命令都被包含 | write=CREATE
" -m gemini-2.5-flash --approval-mode yolo

# 检查生成结果
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ 索引生成成功！${NC}"
    echo ""
    echo "生成的索引文件:"
    ls -lh "$INDEX_DIR"/*.json
    echo ""

    # 验证文件
    echo -e "${YELLOW}验证索引文件...${NC}"
    REQUIRED_FILES=("all-commands.json" "by-category.json" "by-use-case.json" "essential-commands.json" "command-relationships.json")
    ALL_EXIST=true

    for file in "${REQUIRED_FILES[@]}"; do
        if [ -f "$INDEX_DIR/$file" ]; then
            echo -e "${GREEN}✓${NC} $file"
        else
            echo -e "${RED}✗${NC} $file (缺失)"
            ALL_EXIST=false
        fi
    done

    echo ""

    if [ "$ALL_EXIST" = true ]; then
        echo -e "${GREEN}=== 索引更新完成！===${NC}"
        echo ""
        echo "后续步骤:"
        echo "1. 验证生成的索引内容是否正确"
        echo "2. 提交更新: git add .claude/skills/command-guide/index/"
        echo "3. 创建提交: git commit -m \"docs: 更新命令索引\""
        echo "4. 推送更新: git push"
        echo ""
        echo "团队成员执行 'git pull' 后将自动获取最新索引"
    else
        echo -e "${RED}=== 索引更新不完整，请检查错误 ===${NC}"
        exit 1
    fi
else
    echo ""
    echo -e "${RED}✗ 索引生成失败${NC}"
    echo "请检查 gemini 输出的错误信息"
    exit 1
fi
