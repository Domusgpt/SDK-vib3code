#!/bin/bash
# Hook: Red-team test analysis
# Forces critical thinking about whether tests actually cover the changes

TOOL_OUTPUT="$1"

# Detect test pass output
if echo "$TOOL_OUTPUT" | grep -qiE "(tests? pass|✓.*test|Test Files.*passed)"; then
  echo "TEST RESULTS DETECTED - RED TEAM ANALYSIS REQUIRED"
  echo ""
  echo "Before claiming these tests validate your changes, answer:"
  echo ""
  echo "1. WHAT did you change? (browser code, Node code, config, etc.)"
  echo "2. WHAT do these tests actually test? (unit tests, integration, e2e?)"
  echo "3. Do these tests RUN IN A BROWSER or just Node.js?"
  echo "4. If you changed browser code and tests run in Node, say:"
  echo "   'These tests do not verify my browser changes.'"
  echo ""
  echo "Do NOT conflate 'tests pass' with 'my changes work'."
  exit 0
fi
