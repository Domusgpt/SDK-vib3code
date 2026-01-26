/**
 * VIB3+ MCP Module
 * Re-exports MCP server and tools
 */

export { MCPServer } from './MCPServer.js';
export { toolDefinitions, getToolList, getToolNames, getTool, validateToolInput } from './tools.js';

export { default as mcpServer } from './MCPServer.js';
