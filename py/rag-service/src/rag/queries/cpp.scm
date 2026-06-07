; C++ symbol queries — definitions only
(function_definition
  declarator: (function_declarator
    declarator: (identifier) @function))
(function_definition
  declarator: (function_declarator
    declarator: (qualified_identifier) @method))
(struct_specifier name: (type_identifier) @class)
(class_specifier name: (type_identifier) @class)
(enum_specifier name: (type_identifier) @type)
(type_definition
  declarator: (type_identifier) @type)
(namespace_definition name: (namespace_identifier) @module)

