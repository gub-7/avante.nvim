; Rust symbol queries — definitions only
(function_item name: (identifier) @function)
(struct_item name: (type_identifier) @type)
(enum_item name: (type_identifier) @type)
(trait_item name: (type_identifier) @interface)
(type_item name: (type_identifier) @type)
(const_item name: (identifier) @constant)
(static_item name: (identifier) @constant)
(impl_item
  body: (declaration_list
    (function_item name: (identifier) @method)))
(mod_item name: (identifier) @module)

