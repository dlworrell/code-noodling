#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:58c48e7e-5484-4c12-ab02-6ffd525734af

"""Definite-initialization abstract execution of the actual 16-limb C AST.

All secret data is TOP (initialized but unknown). Unknown control/index values
are rejected. The restricted interpreter/parser are trusted verification code.
"""
import argparse
import hashlib
import json
from pathlib import Path
from pycparser import c_ast, c_generator
from check_information_flow import parse
from restricted_c import ROOT

TOP = object()
UNINIT = object()


class InitError(ValueError):
    pass


class Returned(Exception):
    pass


class Array:
    def __init__(self, name, length, initial):
        self.name=name
        self.values=[initial]*length


class Machine:
    def __init__(self):
        self.arrays={name:Array(name,16,TOP if name in ('a','b','modulus') else UNINIT)
                     for name in ('a','b','modulus','out')}
        self.env={'a':self.arrays['a'],'b':self.arrays['b'],'out':self.arrays['out'],
                  'ctx':{'limbs':16,'inverse':TOP,'modulus':self.arrays['modulus']},
                  'a_capacity':16,'b_capacity':16,'out_capacity':16,
                  'CRYPTO_MAX_LIMBS':1024,'EINVAL':22,'EDOM':33,'errno':TOP}
        self.reads=0;self.writes=0;self.iterations=0

    def value(self,node):
        if isinstance(node,c_ast.ID):
            value=self.env.get(node.name,UNINIT)
            if value is UNINIT: raise InitError('uninitialized scalar '+node.name)
            return value
        if isinstance(node,c_ast.Constant):
            return int(node.value.rstrip('uUlL'),0)
        if isinstance(node,c_ast.ArrayRef):
            array,index=self.location(node)
            value=array.values[index]
            if value is UNINIT: raise InitError(f'uninitialized array read {array.name}[{index}]')
            self.reads+=1
            return value
        if isinstance(node,c_ast.StructRef):
            return self.value(node.name)[node.field.name]
        if isinstance(node,c_ast.Cast):
            value=self.value(node.expr)
            if value is TOP: return TOP
            width={'limb_t':64,'dlimb_t':128}[c_generator.CGenerator().visit(node.to_type)]
            return value % (1<<width)
        if isinstance(node,c_ast.UnaryOp):
            if node.op=='sizeof':
                if c_generator.CGenerator().visit(node)!='sizeof(limb_t)': raise InitError('unsupported sizeof')
                return 8
            value=self.value(node.expr)
            if node.op in ('++','p++'):
                if value is TOP: raise InitError('unknown increment')
                self.env[node.expr.name]=value+1
                return value if node.op=='p++' else value+1
            if value is TOP: return TOP
            result={'!':lambda:int(not value),'~':lambda:~value,'-':lambda:-value,'+':lambda:+value}[node.op]()
            if not isinstance(result,int) or not 0<=result<2**31: raise InitError('unsupported public arithmetic range')
            return result
        if isinstance(node,c_ast.BinaryOp):
            left=self.value(node.left)
            if node.op in ('&&','||'):
                if left is TOP: raise InitError('unknown short circuit')
                if node.op=='&&' and not left: return 0
                if node.op=='||' and left: return 1
            right=self.value(node.right)
            if left is TOP or right is TOP: return TOP
            operations={'+':lambda:left+right,'-':lambda:left-right,'*':lambda:left*right,
                        '>>':lambda:left>>right,'<<':lambda:left<<right,
                        '|':lambda:left|right,'&':lambda:left&right,'^':lambda:left^right,
                        '==':lambda:int(left==right),'!=':lambda:int(left!=right),
                        '<':lambda:int(left<right),'>':lambda:int(left>right),
                        '<=':lambda:int(left<=right),'>=':lambda:int(left>=right),
                        '&&':lambda:int(bool(left) and bool(right)),
                        '||':lambda:int(bool(left) or bool(right))}
            result=operations[node.op]()
            if not isinstance(result,int) or not 0<=result<2**31: raise InitError('unsupported public arithmetic range')
            return result
        raise InitError('unsupported expression '+type(node).__name__)

    def location(self,node):
        array=self.value(node.name); index=self.value(node.subscript)
        if not isinstance(array,Array) or index is TOP or not isinstance(index,int):
            raise InitError('unknown address')
        if not 0<=index<len(array.values): raise InitError('out of bounds')
        return array,index

    def execute(self,node):
        if node is None: return
        if isinstance(node,c_ast.Compound):
            for stmt in node.block_items or []: self.execute(stmt)
        elif isinstance(node,c_ast.DeclList):
            for stmt in node.decls: self.execute(stmt)
        elif isinstance(node,c_ast.Decl):
            if isinstance(node.type,c_ast.ArrayDecl):
                size=self.value(node.type.dim)
                if size is TOP or not isinstance(size,int) or not 0<size<=32: raise InitError('unknown array size')
                array=Array(node.name,size,UNINIT)
                self.arrays[node.name]=array;self.env[node.name]=array
            else:
                self.env[node.name]=self.value(node.init) if node.init is not None else UNINIT
        elif isinstance(node,c_ast.Assignment):
            right=self.value(node.rvalue)
            if node.op!='=':
                old=self.value(node.lvalue)
                if old is TOP or right is TOP: right=TOP
                elif node.op=='+=': right=old+right
                else: raise InitError('unsupported assignment')
            if isinstance(node.lvalue,c_ast.ID): self.env[node.lvalue.name]=right
            elif isinstance(node.lvalue,c_ast.ArrayRef):
                array,index=self.location(node.lvalue)
                array.values[index]=right;self.writes+=1
            else: raise InitError('unsupported write')
        elif isinstance(node,c_ast.If):
            condition=self.value(node.cond)
            if condition is TOP: raise InitError('unknown branch')
            self.execute(node.iftrue if condition else node.iffalse)
        elif isinstance(node,c_ast.For):
            self.execute(node.init)
            count=0
            while True:
                condition=self.value(node.cond)
                if condition is TOP: raise InitError('unknown loop bound')
                if not condition: break
                count+=1;self.iterations+=1
                if count>32: raise InitError('unsupported loop extent')
                self.execute(node.stmt);self.value(node.next)
        elif isinstance(node,c_ast.FuncCall):
            if node.name.name!='memset': raise InitError('unsupported call')
            array,fill,size=[self.value(x) for x in node.args.exprs]
            if not isinstance(array,Array) or fill!=0 or size is TOP or size%8 or not 0<=size<=8*len(array.values):
                raise InitError('unsupported memset')
            for i in range(size//8): array.values[i]=0;self.writes+=1
        elif isinstance(node,c_ast.Return):
            result=self.value(node.expr)
            if result!=0: raise InitError('unexpected early return')
            raise Returned()
        else: raise InitError('unsupported statement '+type(node).__name__)


def check(source):
    function=parse(source)
    machine=Machine()
    try:
        machine.execute(function.body)
    except Returned:
        pass
    else:
        raise InitError("missing return")
    for i,value in enumerate(machine.arrays['out'].values):
        if value is UNINIT: raise InitError(f'uninitialized output out[{i}]')
    for name in ('t','difference'):
        if any(value!=0 for value in machine.arrays[name].values): raise InitError('incomplete scratch wipe '+name)
    return {'source_sha256':hashlib.sha256(source.encode()).hexdigest(),
            'limbs':16,'checked_reads':machine.reads,'writes':machine.writes,
            'loop_iterations':machine.iterations,'output_initialized':True,'scratch_arrays_zero_on_return':True,
            'scope':'valid public 16-limb inputs; initialized input/context payloads; TOP represents every secret value',
            'trusted':['restricted abstract interpreter','pycparser','C-to-model mapping; concrete public arithmetic stays in range']}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    source=(ROOT/'crypto_montgomery_ct.c').read_text()
    certificate=check(source)
    controls={
        'missing-zeroing':('memset(t, 0, 2*n * sizeof(limb_t));','memset(t, 0, 0 * sizeof(limb_t));'),
        'missing-difference-elements':('difference[i] = (limb_t)', 'difference[0] = (limb_t)'),
        'missing-output-element':('i < n; ++i) out[i]', 'i < n-1; ++i) out[i]'),
        'missing-carry-initialization':('dlimb_t carry = 0;', 'dlimb_t carry;'),
        'incomplete-wipe':('i < 2*n; ++i) p[i] = 0;', 'i < 2*n-1; ++i) p[i] = 0;')}
    certificate['negative_controls']=[]
    for name,(old,new) in controls.items():
        if old not in source: raise InitError('mutation anchor drift')
        try: check(source.replace(old,new))
        except InitError as error:
            if not ('uninitialized' in str(error) or 'incomplete scratch wipe' in str(error)): raise
            certificate['negative_controls'].append({'name':name,'rejected':str(error)})
        else: raise InitError('missed mutation '+name)
    args.output.write_text(json.dumps(certificate,indent=2)+'\n')
    print('PASS: definite initialization and scratch wiping; 5 mutations rejected')
