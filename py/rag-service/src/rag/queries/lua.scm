; Lua symbol queries — definitions only
(function_declaration name: (identifier) @function)
(function_declaration name: (dot_index_expression) @method)
(function_declaration name: (method_index_expression) @method)
(local_function name: (identifier) @function)
(assignment_statement
  (variable_list (identifier) @variable)
  (expression_list (function_definition)))

