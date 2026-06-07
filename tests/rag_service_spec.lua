local mock = require("luassert.mock")
local match = require("luassert.match")

describe("RagService", function()
  local RagService
  local Config_mock

  before_each(function()
    -- Load the module before each test
    RagService = require("avante.rag_service")

    -- Setup common mocks
    Config_mock = mock(require("avante.config"), true)
    Config_mock.rag_service = { host_mount = "/home/user" }
  end)

  after_each(function()
    -- Clean up after each test
    package.loaded["avante.rag_service"] = nil
    mock.revert(Config_mock)
  end)

  describe("URI conversion functions", function()
    it("should convert URIs between host and container formats", function()
      -- Test both directions of conversion
      local host_uri = "file:///home/user/project/file.txt"
      local container_uri = "file:///host/project/file.txt"

      -- Host to container
      local result1 = RagService.to_container_uri(host_uri)
      assert.equals(container_uri, result1)

      -- Container to host
      local result2 = RagService.to_local_uri(container_uri)
      assert.equals(host_uri, result2)
    end)
  end)

  -- ---------------------------------------------------------------------------
  -- Increment 12 — new routing kwargs are serialised into the request body
  -- ---------------------------------------------------------------------------
  describe("Increment-12 routing kwargs", function()
    local curl_mock
    local captured_body

    before_each(function()
      -- Capture the JSON body that rag_retrieve / rag_search / etc. send via curl.
      captured_body = nil
      curl_mock = mock(require("plenary.curl"), true)
      curl_mock.post.invokes(function(_, opts)
        captured_body = vim.json.decode(opts.body)
        -- Invoke the callback with a synthetic 200 response so on_complete is called.
        opts.callback({
          status = 200,
          body = vim.json.encode({ spans = {}, sources = {}, citations = {}, token_estimate = 0 }),
        })
      end)
    end)

    after_each(function()
      package.loaded["plenary.curl"] = nil
      mock.revert(curl_mock)
      captured_body = nil
    end)

    it("rag_retrieve serialises mode, shadow, purpose, parent_request_id into body", function()
      local query_body = {
        query = "what does foo() return?",
        base_uri = "file:///home/user/project",
        top_k = 5,
      }
      local opts = {
        mode = "exact",
        shadow = true,
        purpose = "agentic",
        parent_request_id = "parent-xyz",
      }

      RagService.rag_retrieve(query_body, function() end, opts)

      assert.is_not_nil(captured_body, "curl.post was not called")
      assert.equals("exact", captured_body.mode)
      assert.is_true(captured_body.shadow)
      assert.equals("agentic", captured_body.purpose)
      assert.equals("parent-xyz", captured_body.parent_request_id)
    end)

    it("rag_retrieve omits routing fields when opts is nil", function()
      local query_body = {
        query = "what does foo() return?",
        base_uri = "file:///home/user/project",
        top_k = 5,
      }

      RagService.rag_retrieve(query_body, function() end, nil)

      assert.is_not_nil(captured_body, "curl.post was not called")
      assert.is_nil(captured_body.mode)
      assert.is_nil(captured_body.shadow)
      assert.is_nil(captured_body.purpose)
      assert.is_nil(captured_body.parent_request_id)
    end)

    it("rag_search serialises mode into body", function()
      local query_body = {
        query = "FooBar class definition",
        base_uri = "file:///home/user/project",
        top_k = 3,
      }

      RagService.rag_search(query_body, function() end, { mode = "hybrid" })

      assert.is_not_nil(captured_body, "curl.post was not called")
      assert.equals("hybrid", captured_body.mode)
    end)

    it("rag_context serialises purpose into body", function()
      local query_body = {
        query = "explain the retry logic",
        base_uri = "file:///home/user/project",
        top_k = 5,
      }

      RagService.rag_context(query_body, function() end, { purpose = "context" })

      assert.is_not_nil(captured_body, "curl.post was not called")
      assert.equals("context", captured_body.purpose)
    end)

    it("rag_agentic_retrieve serialises parent_request_id into body", function()
      local query_body = {
        query = "list all exported functions",
        base_uri = "file:///home/user/project",
        top_k = 5,
      }

      RagService.rag_agentic_retrieve(query_body, function() end, { parent_request_id = "root-001" })

      assert.is_not_nil(captured_body, "curl.post was not called")
      assert.equals("root-001", captured_body.parent_request_id)
    end)
  end)
end)
