; C symbol queries — definitions only
(function_definition
  declarator: (function_declarator
    declarator: (identifier) @function))
(struct_specifier name: (type_identifier) @class)
(enum_specifier name: (type_identifier) @type)
(type_definition
  declarator: (type_identifier) @type)

