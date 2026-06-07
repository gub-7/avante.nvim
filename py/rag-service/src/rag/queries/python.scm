; Python symbol queries — definitions only
(function_definition name: (identifier) @function)
(class_definition name: (identifier) @class)
(assignment
  left: (identifier) @constant
  (#match? @constant "^[A-Z_][A-Z0-9_]*$"))

