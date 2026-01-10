#!/bin/bash
# Hook: Read recently modified GitHub docs
# Triggers on "please read the docs" + GitHub URL

USER_MESSAGE="$1"

# Detect request to read docs with GitHub link
if echo "$USER_MESSAGE" | grep -qiE "please read.*(docs|documentation)"; then
  if echo "$USER_MESSAGE" | grep -qoE "github\.com/[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+"; then
    REPO=$(echo "$USER_MESSAGE" | grep -oE "github\.com/[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+" | head -1)
    echo "GITHUB DOCS REQUEST DETECTED"
    echo ""
    echo "Repository: $REPO"
    echo ""
    echo "You MUST:"
    echo "1. Use GitHub API or WebFetch to list recently modified files"
    echo "2. Read ALL .md files modified in the last 7 days"
    echo "3. Read CLAUDE.md if it exists"
    echo "4. Read any file mentioned in the user's message"
    echo ""
    echo "Do this BEFORE responding to the user's question."
    exit 0
  fi
fi
