#!/bin/bash
# Hook: Force verification of deployed code after git push
# Triggers when git push output is detected

TOOL_OUTPUT="$1"

# Detect git push to deployment branches
if echo "$TOOL_OUTPUT" | grep -qiE "(-> (main|master|gh-pages|claude/))|pushed to origin"; then
  echo "DEPLOYMENT DETECTED - VERIFICATION REQUIRED"
  echo ""
  echo "You just pushed code. Before claiming success, you MUST:"
  echo "1. Use WebFetch to load the deployed URL"
  echo "2. Check for JavaScript errors in the response"
  echo "3. Verify the page actually renders (not stuck on loading)"
  echo ""
  echo "Do NOT say 'done' or 'success' until WebFetch confirms the site works."
  exit 0
fi
