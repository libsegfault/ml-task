from tree_sitter_language_pack import get_language, get_parser

lang = get_language('go')

query ="""
(function_declaration
    (identifier) @name
    (block) @body
)

(method_declaration
    (field_identifier) @name
    (block) @body
)

(comment) @comm
"""

extra_id = 'func <extra_id_0>'
