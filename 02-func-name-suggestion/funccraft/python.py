from tree_sitter_language_pack import get_language, get_parser

lang = get_language('python')

query ="""
(function_definition
    name: (identifier) @name
    body: (block) @body
)

(expression_statement
    (string) @comm
)

(comment) @comm
"""

extra_id = 'def <extra_id_0>():'
