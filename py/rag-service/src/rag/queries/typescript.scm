; TypeScript symbol queries — definitions only
(function_declaration name: (identifier) @function)
(method_definition key: (property_identifier) @method)
(class_declaration name: (type_identifier) @class)
(interface_declaration name: (type_identifier) @interface)
(type_alias_declaration name: (type_identifier) @type)
(generator_function_declaration name: (identifier) @function)
(enum_declaration name: (identifier) @type)
(abstract_class_declaration name: (type_identifier) @class)
(lexical_declaration
  (variable_declarator
    name: (identifier) @constant
    value: [(arrow_function) (function_expression)])
  (#match? @constant "^[A-Z_][A-Z0-9_]*$"))

