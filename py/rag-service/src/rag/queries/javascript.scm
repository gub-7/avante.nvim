; JavaScript symbol queries — definitions only
(function_declaration name: (identifier) @function)
(method_definition key: (property_identifier) @method)
(class_declaration name: (identifier) @class)
(generator_function_declaration name: (identifier) @function)
(lexical_declaration
  (variable_declarator
    name: (identifier) @constant
    value: [(arrow_function) (function_expression)])
  (#match? @constant "^[A-Z_][A-Z0-9_]*$"))

