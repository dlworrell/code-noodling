#!/usr/bin/env python3
"""Conservative source-level information-flow checker for this restricted kernel.

Least fixed point over scalar dependencies. Input and scratch array contents are
always secret; context contents and addresses are public. All control decisions,
indices, array lengths, and memset lengths must be public. The checker/parser
are trusted, not themselves formally verified. No timing or compiler theorem.
"""
import argparse
import hashlib
import json
from pathlib import Path
from pycparser import c_ast, c_parser, c_generator
from restricted_c import PRELUDE, Validate, ROOT
import re

PUBLIC = {'out','a','b','ctx','out_capacity','a_capacity','b_capacity','CRYPTO_MAX_LIMBS',
          'EINVAL','EDOM','errno','t','difference','p'}
SECRET_ARRAYS = {'a','b','out','t','difference','p'}


class FlowError(ValueError):
    pass


def parse(source):
    # Remove includes and comments without changing literal contents. Literals
    # other than integer constants are rejected by the restricted checker.
    text = re.sub(r'^\s*#include[^\n]*', '', source, flags=re.M)
    tokens = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|/\*.*?\*/|//[^\n]*',re.S)
    text = tokens.sub(lambda m: '' if m.group().startswith(('/',)) else m.group(),text)
    if '#' in text:
        raise FlowError('Unsupported preprocessor directive')
    ast = c_parser.CParser().parse(PRELUDE+text)
    matches = [x for x in ast.ext if isinstance(x,c_ast.FuncDef) and x.decl.name == 'crypto_montgomery_mul_ct']
    if len(matches) != 1:
        raise FlowError('Expected one candidate definition')
    function = matches[0]
    Validate().visit(function)
    return function


class Nodes(c_ast.NodeVisitor):
    def __init__(self):
        self.items=[]
    def generic_visit(self,node):
        self.items.append(node)
        super().generic_visit(node)


def check(source):
    function = parse(source)
    visitor=Nodes();visitor.visit(function)
    nodes=visitor.items
    declared={n.name for n in nodes if isinstance(n,c_ast.Decl)}
    names=PUBLIC | declared
    secret=set()
    generator=c_generator.CGenerator()

    def dependencies(node):
        if node is None:
            return set()
        if isinstance(node,c_ast.Constant):
            if node.type not in ('int','unsigned int','long int','unsigned long int','long long int','unsigned long long int'):
                raise FlowError('Only integer constants supported')
            return set()
        if isinstance(node,c_ast.ID):
            if node.name not in names:
                raise FlowError('Unknown value '+node.name)
            return {node.name} if node.name in secret else set()
        if isinstance(node,c_ast.ArrayRef):
            label = generator.visit(node.name)
            address = dependencies(node.subscript)
            if label == 'ctx->modulus':
                return address
            if label not in SECRET_ARRAYS:
                raise FlowError('Unknown memory object '+label)
            return {'contents:'+label} | address | dependencies(node.name)
        if isinstance(node,c_ast.StructRef):
            if generator.visit(node) not in ('ctx->limbs','ctx->inverse'):
                raise FlowError('Unsupported scalar context access')
            return set()
        if isinstance(node,c_ast.Cast):
            return dependencies(node.expr)
        if isinstance(node,c_ast.UnaryOp):
            if node.op == 'sizeof':
                # The candidate uses sizeof(limb_t), not a VLA expression.
                if generator.visit(node) != 'sizeof(limb_t)':
                    raise FlowError('Unsupported sizeof')
                return set()
            return dependencies(node.expr)
        if isinstance(node,c_ast.BinaryOp):
            return dependencies(node.left) | dependencies(node.right)
        if isinstance(node,c_ast.TernaryOp):
            return dependencies(node.cond) | dependencies(node.iftrue) | dependencies(node.iffalse)
        raise FlowError('Unsupported value expression '+type(node).__name__)

    # Every scalar write contributes a constraint; name collisions across scopes
    # conservatively merge labels. No path-sensitive declassification is allowed.
    changed=True
    while changed:
        changed=False
        for node in nodes:
            target=None; rhs=None
            if isinstance(node,c_ast.Decl) and node.init is not None:
                target=node.name;rhs=node.init
            elif isinstance(node,c_ast.Assignment):
                if isinstance(node.lvalue,c_ast.ID):
                    target=node.lvalue.name;rhs=node.rvalue
                elif isinstance(node.lvalue,c_ast.StructRef) or (isinstance(node.lvalue,c_ast.ArrayRef) and isinstance(node.lvalue.name,c_ast.StructRef)):
                    raise FlowError('Context writes are outside the model')
            if target is not None and dependencies(rhs) and target not in secret:
                secret.add(target);changed=True

    obligations=[]
    def public(node,kind):
        deps=dependencies(node)
        if deps:
            raise FlowError(kind+' depends on '+','.join(sorted(deps))+': '+generator.visit(node))
        obligations.append({'kind':kind,'expression':generator.visit(node),'dependencies':[]})

    for node in nodes:
        if isinstance(node,(c_ast.If,c_ast.For,c_ast.TernaryOp)):
            public(node.cond,'control')
        elif isinstance(node,c_ast.BinaryOp) and node.op in ('&&','||'):
            public(node.left,'short-circuit')
        elif isinstance(node,c_ast.ArrayRef):
            public(node.subscript,'array-index')
            if isinstance(node.name,c_ast.ID):
                public(node.name,'array-base')
        elif isinstance(node,c_ast.ArrayDecl):
            public(node.dim,'array-length')
        elif isinstance(node,c_ast.FuncCall):
            # Validate already restricts calls to memset(t,0,length).
            for expression in node.args.exprs:
                public(expression,'memset-argument')
        elif isinstance(node,c_ast.Decl) and isinstance(node.type,c_ast.PtrDecl) and node.init is not None:
            public(node.init,'pointer-initialization')
        elif isinstance(node,c_ast.Assignment) and isinstance(node.lvalue,c_ast.ID) and node.lvalue.name == 'p':
            public(node.rvalue,'pointer-assignment')
    return {'source_sha256':hashlib.sha256(source.encode()).hexdigest(),
            'secret_scalars':sorted(secret),'obligations':obligations,
            'scope':'Restricted source-level branch/address noninterference; assumes defined C execution',
            'trusted':['this checker','pycparser 2.23','restricted C model and public/secret contract']}


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=ROOT/'crypto_montgomery_ct.c')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    certificate=check(args.source.read_text())
    # Mutation controls must be rejected by the actual analysis, not just syntax.
    source=args.source.read_text()
    anchor='    size_t n = ctx->limbs;'
    controls={'branch': '    if (a[0] & 1) { errno = 0; }\n',
              'address': '    out[0] = a[a[0] & 15];\n',
              'indirect-scalar': '    limb_t hidden = a[0]; if (hidden) { errno = 0; }\n',
              'loop-bound': '    for (size_t k = 0; k < a[0]; ++k) { out[0] = 0; }\n'}
    certificate['negative_controls']=[]
    if source.count(anchor) != 1:
        raise FlowError('Mutation anchor drift')
    for name,leak in controls.items():
        try:
            check(source.replace(anchor,leak+anchor))
        except FlowError as error:
            if 'depends on' not in str(error):
                raise
            certificate['negative_controls'].append({'name':name,'rejected':str(error)})
        else:
            raise FlowError('Missed deliberate leak '+name)
    args.output.write_text(json.dumps(certificate,indent=2)+'\n')
    print(f"PASS: {len(certificate['obligations'])} public-flow obligations; {len(controls)} deliberate leaks rejected")
