"""Fail-closed syntax restrictions for the source information-flow checker.

This parser/model is trusted verification code, not a verified C frontend.
"""
from pathlib import Path
from pycparser import c_ast, c_generator

ROOT = Path(__file__).resolve().parents[1]
PRELUDE = '''typedef unsigned long size_t;
typedef unsigned long uintptr_t;
typedef unsigned long long limb_t;
typedef unsigned __int128 dlimb_t;
typedef struct { size_t limbs; limb_t inverse; limb_t modulus[1024]; } crypto_montgomery_ctx;
'''


class Validate(c_ast.NodeVisitor):
    ALLOWED = {getattr(c_ast, name) for name in '''FuncDef Decl FuncDecl ParamList PtrDecl TypeDecl IdentifierType Compound If Return Constant ID BinaryOp UnaryOp Assignment For DeclList ArrayDecl ArrayRef Cast Typename StructRef FuncCall ExprList TernaryOp'''.split()}

    def generic_visit(self, node):
        if type(node) not in self.ALLOWED:
            raise ValueError(f'Unsupported syntax: {type(node).__name__}')
        super().generic_visit(node)

    def visit_PtrDecl(self, node):
        if not isinstance(node.type, c_ast.TypeDecl) or node.type.declname not in ('out', 'a', 'b', 'ctx', 'p'):
            raise ValueError('Unsupported pointer declaration')
        self.generic_visit(node)

    def visit_Decl(self, node):
        if isinstance(node.type, c_ast.PtrDecl) and node.init is not None:
            if node.name != 'p' or not isinstance(node.init, c_ast.ID) or node.init.name != 't':
                raise ValueError('Unsupported pointer initialization')
        self.generic_visit(node)

    def visit_Assignment(self, node):
        if isinstance(node.lvalue, c_ast.ID) and node.lvalue.name in ('a', 'b', 'out', 'ctx', 'p'):
            if node.op != '=' or node.lvalue.name != 'p' or not isinstance(node.rvalue, c_ast.ID) or node.rvalue.name not in ('t', 'difference'):
                raise ValueError('Unsupported pointer assignment')
        self.generic_visit(node)

    def visit_ArrayRef(self, node):
        simple = isinstance(node.name, c_ast.ID) and node.name.name in ('a', 'b', 'out', 't', 'difference', 'p')
        modulus = isinstance(node.name, c_ast.StructRef) and c_generator.CGenerator().visit(node.name) == 'ctx->modulus'
        if not (simple or modulus):
            raise ValueError('Unsupported array base')
        self.generic_visit(node)

    def visit_StructRef(self, node):
        if c_generator.CGenerator().visit(node) not in ('ctx->limbs', 'ctx->inverse', 'ctx->modulus'):
            raise ValueError('Unsupported struct access')
        self.generic_visit(node)

    def visit_UnaryOp(self, node):
        if node.op not in ('!', '~', '-', '+', 'sizeof', 'p++', '++'):
            raise ValueError('Unsupported unary/pointer operation')
        if node.op in ('p++', '++') and (not isinstance(node.expr, c_ast.ID) or node.expr.name not in ('i', 'j', 'k')):
            raise ValueError('Only loop counter increments are supported')
        self.generic_visit(node)

    def visit_BinaryOp(self, node):
        if node.op not in ('+', '-', '*', '>>', '<<', '&', '|', '^', '==', '!=', '<', '>', '<=', '>=', '&&', '||'):
            raise ValueError('Unsupported operator')
        self.generic_visit(node)

    def visit_Cast(self, node):
        if c_generator.CGenerator().visit(node.to_type) not in ('limb_t', 'dlimb_t'):
            raise ValueError('Unsupported cast')
        self.generic_visit(node)

    def visit_FuncCall(self, node):
        if not isinstance(node.name, c_ast.ID) or node.name.name != 'memset' or len(node.args.exprs) != 3:
            raise ValueError('Unsupported function call')
        if c_generator.CGenerator().visit(node.args.exprs[0]) != 't' or c_generator.CGenerator().visit(node.args.exprs[1]) != '0':
            raise ValueError('Only zero initialization of t is modeled')
        self.generic_visit(node)

