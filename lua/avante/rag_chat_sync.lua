---@mod avante-rag-chat-sync avante RAG chat-history sync
---@brief [[
---
--- Debounced synchronisation of chat history saves/deletes to the RAG
--- service chat-history endpoints.  Loaded lazily by path.lua hooks;
--- requires ``Config.rag_service.index_chat_history = true``.
---
---@brief ]]

local M = {}

local rag = require("avante.rag_service")
local Config = require("avante.config")
local Utils = require("avante.utils")

-- Pending debounce timers, keyed by bufnr
local pending = {}
local DEBOUNCE_MS = 250

--- Build a ChatTurnUpsert-compatible table from a history object.
---@param history avante.ChatHistory
---@param project_root string
---@return table
local function build_turn(history, project_root)
  local messages = {}
  for _, m in ipairs(history.messages or {}) do
    table.insert(messages, {
      role = m.role or "user",
      content = m.content or "",
      timestamp = m.timestamp or tostring(os.time()),
      tool_name = m.tool_name,
    })
  end
  return {
    -- to_container_uri translation is performed inside rag_service.lua
    base_uri = "file://" .. project_root,
    chat_id = (history.filename or ""):gsub("%.json$", ""),
    title = history.title,
    project_root = project_root,
    messages = messages,
    updated_at = os.date("!%Y-%m-%dT%H:%M:%SZ"),
  }
end

--- Called by path.lua after a chat history file is saved.
---
--- Debounces rapid successive saves (e.g. during streaming) and pushes the
--- resulting turn to the RAG service chat-history upsert endpoint.
---
---@param bufnr integer   Neovim buffer number owning this history.
---@param history avante.ChatHistory  The history table that was saved.
function M.on_save(bufnr, history)
  if not (Config.rag_service and Config.rag_service.index_chat_history) then return end

  -- Cancel any pending timer for this buffer
  if pending[bufnr] then
    pending[bufnr]:stop()
    pending[bufnr] = nil
  end

  pending[bufnr] = vim.defer_fn(function()
    pending[bufnr] = nil
    local root = Utils.root.get({ buf = bufnr })
    if not root or root == "" then return end
    rag.chat_history_upsert(build_turn(history, root))
  end, DEBOUNCE_MS)
end

--- Called by path.lua after a chat history file is deleted.
---
--- Notifies the RAG service to remove the corresponding chat turn.
---
---@param bufnr integer  Neovim buffer number owning the history.
---@param filename string  The filename (basename) of the deleted history file.
function M.on_delete(bufnr, filename)
  if not (Config.rag_service and Config.rag_service.index_chat_history) then return end

  local root = Utils.root.get({ buf = bufnr })
  if not root or root == "" then return end

  local chat_id = filename:gsub("%.json$", "")
  rag.chat_history_delete("file://" .. root, chat_id)
end

return M

