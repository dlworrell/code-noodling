
build/ct-local/candidate.o:     file format elf64-x86-64


Disassembly of section .text:

0000000000000000 <crypto_montgomery_ctx_init>:
   0:	f3 0f 1e fa          	endbr64
   4:	41 54                	push   r12
   6:	55                   	push   rbp
   7:	53                   	push   rbx
   8:	48 85 ff             	test   rdi,rdi
   b:	0f 84 1e 01 00 00    	je     12f <crypto_montgomery_ctx_init+0x12f>
  11:	48 89 f5             	mov    rbp,rsi
  14:	48 85 f6             	test   rsi,rsi
  17:	0f 84 12 01 00 00    	je     12f <crypto_montgomery_ctx_init+0x12f>
  1d:	48 8d 42 ff          	lea    rax,[rdx-0x1]
  21:	49 89 d4             	mov    r12,rdx
  24:	48 3d ff 03 00 00    	cmp    rax,0x3ff
  2a:	0f 87 ff 00 00 00    	ja     12f <crypto_montgomery_ctx_init+0x12f>
  30:	49 89 f8             	mov    r8,rdi
  33:	31 d2                	xor    edx,edx
  35:	49 83 fc 01          	cmp    r12,0x1
  39:	74 35                	je     70 <crypto_montgomery_ctx_init+0x70>
  3b:	4a 8d 0c e6          	lea    rcx,[rsi+r12*8]
  3f:	48 8d 46 08          	lea    rax,[rsi+0x8]
  43:	31 d2                	xor    edx,edx
  45:	48 89 ce             	mov    rsi,rcx
  48:	48 29 c6             	sub    rsi,rax
  4b:	83 e6 08             	and    esi,0x8
  4e:	74 10                	je     60 <crypto_montgomery_ctx_init+0x60>
  50:	48 8b 10             	mov    rdx,QWORD PTR [rax]
  53:	48 8d 45 10          	lea    rax,[rbp+0x10]
  57:	48 39 c8             	cmp    rax,rcx
  5a:	74 14                	je     70 <crypto_montgomery_ctx_init+0x70>
  5c:	0f 1f 40 00          	nop    DWORD PTR [rax+0x0]
  60:	48 0b 10             	or     rdx,QWORD PTR [rax]
  63:	48 83 c0 10          	add    rax,0x10
  67:	48 0b 50 f8          	or     rdx,QWORD PTR [rax-0x8]
  6b:	48 39 c8             	cmp    rax,rcx
  6e:	75 f0                	jne    60 <crypto_montgomery_ctx_init+0x60>
  70:	48 8b 75 00          	mov    rsi,QWORD PTR [rbp+0x0]
  74:	48 89 f3             	mov    rbx,rsi
  77:	83 e3 01             	and    ebx,0x1
  7a:	0f 84 9d 00 00 00    	je     11d <crypto_montgomery_ctx_init+0x11d>
  80:	48 83 fe 01          	cmp    rsi,0x1
  84:	75 09                	jne    8f <crypto_montgomery_ctx_init+0x8f>
  86:	48 85 d2             	test   rdx,rdx
  89:	0f 84 8e 00 00 00    	je     11d <crypto_montgomery_ctx_init+0x11d>
  8f:	b8 06 00 00 00       	mov    eax,0x6
  94:	bf 02 00 00 00       	mov    edi,0x2
  99:	0f 1f 80 00 00 00 00 	nop    DWORD PTR [rax+0x0]
  a0:	48 89 f1             	mov    rcx,rsi
  a3:	48 89 fa             	mov    rdx,rdi
  a6:	48 0f af cb          	imul   rcx,rbx
  aa:	48 29 ca             	sub    rdx,rcx
  ad:	48 0f af da          	imul   rbx,rdx
  b1:	83 e8 01             	sub    eax,0x1
  b4:	75 ea                	jne    a0 <crypto_montgomery_ctx_init+0xa0>
  b6:	4c 89 c7             	mov    rdi,r8
  b9:	ba 10 20 00 00       	mov    edx,0x2010
  be:	31 f6                	xor    esi,esi
  c0:	48 f7 db             	neg    rbx
  c3:	e8 00 00 00 00       	call   c8 <crypto_montgomery_ctx_init+0xc8>	c4: R_X86_64_PLT32	memset-0x4
  c8:	48 8b 55 00          	mov    rdx,QWORD PTR [rbp+0x0]
  cc:	4a 8d 0c e5 00 00 00 00 	lea    rcx,[r12*8+0x0]
  d4:	49 89 c0             	mov    r8,rax
  d7:	48 8d 40 10          	lea    rax,[rax+0x10]
  db:	49 89 50 10          	mov    QWORD PTR [r8+0x10],rdx
  df:	49 8d 78 18          	lea    rdi,[r8+0x18]
  e3:	89 ca                	mov    edx,ecx
  e5:	48 8b 74 15 f8       	mov    rsi,QWORD PTR [rbp+rdx*1-0x8]
  ea:	48 83 e7 f8          	and    rdi,0xfffffffffffffff8
  ee:	48 29 f8             	sub    rax,rdi
  f1:	49 89 74 10 08       	mov    QWORD PTR [r8+rdx*1+0x8],rsi
  f6:	01 c1                	add    ecx,eax
  f8:	48 89 ee             	mov    rsi,rbp
  fb:	48 29 c6             	sub    rsi,rax
  fe:	c1 e9 03             	shr    ecx,0x3
 101:	f3 48 a5             	rep movs QWORD PTR es:[rdi],QWORD PTR ds:[rsi]
 104:	49 89 58 08          	mov    QWORD PTR [r8+0x8],rbx
 108:	4d 89 20             	mov    QWORD PTR [r8],r12
 10b:	e8 00 00 00 00       	call   110 <crypto_montgomery_ctx_init+0x110>	10c: R_X86_64_PLT32	__errno_location-0x4
 110:	c7 00 00 00 00 00    	mov    DWORD PTR [rax],0x0
 116:	31 c0                	xor    eax,eax
 118:	5b                   	pop    rbx
 119:	5d                   	pop    rbp
 11a:	41 5c                	pop    r12
 11c:	c3                   	ret
 11d:	e8 00 00 00 00       	call   122 <crypto_montgomery_ctx_init+0x122>	11e: R_X86_64_PLT32	__errno_location-0x4
 122:	c7 00 21 00 00 00    	mov    DWORD PTR [rax],0x21
 128:	b8 ff ff ff ff       	mov    eax,0xffffffff
 12d:	eb e9                	jmp    118 <crypto_montgomery_ctx_init+0x118>
 12f:	e8 00 00 00 00       	call   134 <crypto_montgomery_ctx_init+0x134>	130: R_X86_64_PLT32	__errno_location-0x4
 134:	c7 00 16 00 00 00    	mov    DWORD PTR [rax],0x16
 13a:	eb ec                	jmp    128 <crypto_montgomery_ctx_init+0x128>
 13c:	0f 1f 40 00          	nop    DWORD PTR [rax+0x0]

0000000000000140 <crypto_montgomery_mul_ct>:
 140:	f3 0f 1e fa          	endbr64
 144:	55                   	push   rbp
 145:	48 89 e5             	mov    rbp,rsp
 148:	41 57                	push   r15
 14a:	41 56                	push   r14
 14c:	41 55                	push   r13
 14e:	49 89 d5             	mov    r13,rdx
 151:	41 54                	push   r12
 153:	53                   	push   rbx
 154:	48 83 ec 48          	sub    rsp,0x48
 158:	48 89 7d a8          	mov    QWORD PTR [rbp-0x58],rdi
 15c:	4c 8b 75 10          	mov    r14,QWORD PTR [rbp+0x10]
 160:	64 48 8b 14 25 28 00 00 00 	mov    rdx,QWORD PTR fs:0x28
 169:	48 89 55 c8          	mov    QWORD PTR [rbp-0x38],rdx
 16d:	31 d2                	xor    edx,edx
 16f:	4d 85 f6             	test   r14,r14
 172:	0f 84 ff 02 00 00    	je     477 <crypto_montgomery_mul_ct+0x337>
 178:	48 85 ff             	test   rdi,rdi
 17b:	0f 84 f6 02 00 00    	je     477 <crypto_montgomery_mul_ct+0x337>
 181:	4d 85 ed             	test   r13,r13
 184:	0f 84 ed 02 00 00    	je     477 <crypto_montgomery_mul_ct+0x337>
 18a:	4d 85 c0             	test   r8,r8
 18d:	0f 84 e4 02 00 00    	je     477 <crypto_montgomery_mul_ct+0x337>
 193:	4d 8b 26             	mov    r12,QWORD PTR [r14]
 196:	49 8d 54 24 ff       	lea    rdx,[r12-0x1]
 19b:	48 81 fa ff 03 00 00 	cmp    rdx,0x3ff
 1a2:	0f 87 cf 02 00 00    	ja     477 <crypto_montgomery_mul_ct+0x337>
 1a8:	4c 39 c9             	cmp    rcx,r9
 1ab:	49 0f 47 c9          	cmova  rcx,r9
 1af:	48 39 f1             	cmp    rcx,rsi
 1b2:	48 0f 47 ce          	cmova  rcx,rsi
 1b6:	4c 39 e1             	cmp    rcx,r12
 1b9:	0f 82 b8 02 00 00    	jb     477 <crypto_montgomery_mul_ct+0x337>
 1bf:	4c 89 e2             	mov    rdx,r12
 1c2:	48 89 e1             	mov    rcx,rsp
 1c5:	48 c1 e2 04          	shl    rdx,0x4
 1c9:	48 89 d6             	mov    rsi,rdx
 1cc:	48 89 d0             	mov    rax,rdx
 1cf:	48 81 e6 00 f0 ff ff 	and    rsi,0xfffffffffffff000
 1d6:	48 29 f1             	sub    rcx,rsi
 1d9:	48 39 cc             	cmp    rsp,rcx
 1dc:	74 15                	je     1f3 <crypto_montgomery_mul_ct+0xb3>
 1de:	48 81 ec 00 10 00 00 	sub    rsp,0x1000
 1e5:	48 83 8c 24 f8 0f 00 00 00 	or     QWORD PTR [rsp+0xff8],0x0
 1ee:	48 39 cc             	cmp    rsp,rcx
 1f1:	75 eb                	jne    1de <crypto_montgomery_mul_ct+0x9e>
 1f3:	25 ff 0f 00 00       	and    eax,0xfff
 1f8:	48 29 c4             	sub    rsp,rax
 1fb:	48 85 c0             	test   rax,rax
 1fe:	75 3d                	jne    23d <crypto_montgomery_mul_ct+0xfd>
 200:	4a 8d 04 e5 00 00 00 00 	lea    rax,[r12*8+0x0]
 208:	48 89 e6             	mov    rsi,rsp
 20b:	49 89 e1             	mov    r9,rsp
 20e:	48 89 45 a0          	mov    QWORD PTR [rbp-0x60],rax
 212:	48 83 c0 0f          	add    rax,0xf
 216:	48 89 c1             	mov    rcx,rax
 219:	48 25 00 f0 ff ff    	and    rax,0xfffffffffffff000
 21f:	48 83 e1 f0          	and    rcx,0xfffffffffffffff0
 223:	48 29 c6             	sub    rsi,rax
 226:	48 39 f4             	cmp    rsp,rsi
 229:	74 1a                	je     245 <crypto_montgomery_mul_ct+0x105>
 22b:	48 81 ec 00 10 00 00 	sub    rsp,0x1000
 232:	48 83 8c 24 f8 0f 00 00 00 	or     QWORD PTR [rsp+0xff8],0x0
 23b:	eb e9                	jmp    226 <crypto_montgomery_mul_ct+0xe6>
 23d:	48 83 4c 04 f8 00    	or     QWORD PTR [rsp+rax*1-0x8],0x0
 243:	eb bb                	jmp    200 <crypto_montgomery_mul_ct+0xc0>
 245:	81 e1 ff 0f 00 00    	and    ecx,0xfff
 24b:	48 29 cc             	sub    rsp,rcx
 24e:	48 85 c9             	test   rcx,rcx
 251:	0f 85 32 02 00 00    	jne    489 <crypto_montgomery_mul_ct+0x349>
 257:	48 89 d1             	mov    rcx,rdx
 25a:	31 f6                	xor    esi,esi
 25c:	4c 89 cf             	mov    rdi,r9
 25f:	4c 89 45 b8          	mov    QWORD PTR [rbp-0x48],r8
 263:	48 89 65 b0          	mov    QWORD PTR [rbp-0x50],rsp
 267:	31 db                	xor    ebx,ebx
 269:	e8 00 00 00 00       	call   26e <crypto_montgomery_mul_ct+0x12e>	26a: R_X86_64_PLT32	__memset_chk-0x4
 26e:	4c 8b 45 b8          	mov    r8,QWORD PTR [rbp-0x48]
 272:	49 89 c7             	mov    r15,rax
 275:	49 89 c3             	mov    r11,rax
 278:	49 89 c2             	mov    r10,rax
 27b:	0f 1f 44 00 00       	nop    DWORD PTR [rax+rax*1+0x0]
 280:	4d 8b 4c dd 00       	mov    r9,QWORD PTR [r13+rbx*8+0x0]
 285:	31 c9                	xor    ecx,ecx
 287:	31 f6                	xor    esi,esi
 289:	31 ff                	xor    edi,edi
 28b:	0f 1f 44 00 00       	nop    DWORD PTR [rax+rax*1+0x0]
 290:	4c 89 c8             	mov    rax,r9
 293:	49 f7 24 c8          	mul    QWORD PTR [r8+rcx*8]
 297:	49 03 04 cb          	add    rax,QWORD PTR [r11+rcx*8]
 29b:	48 83 d2 00          	adc    rdx,0x0
 29f:	48 01 c6             	add    rsi,rax
 2a2:	48 11 d7             	adc    rdi,rdx
 2a5:	49 89 34 cb          	mov    QWORD PTR [r11+rcx*8],rsi
 2a9:	48 83 c1 01          	add    rcx,0x1
 2ad:	48 89 fe             	mov    rsi,rdi
 2b0:	31 ff                	xor    edi,edi
 2b2:	49 39 cc             	cmp    r12,rcx
 2b5:	75 d9                	jne    290 <crypto_montgomery_mul_ct+0x150>
 2b7:	48 83 c3 01          	add    rbx,0x1
 2bb:	4b 89 34 e3          	mov    QWORD PTR [r11+r12*8],rsi
 2bf:	49 83 c3 08          	add    r11,0x8
 2c3:	49 39 dc             	cmp    r12,rbx
 2c6:	75 b8                	jne    280 <crypto_montgomery_mul_ct+0x140>
 2c8:	49 8b 46 08          	mov    rax,QWORD PTR [r14+0x8]
 2cc:	4d 89 d1             	mov    r9,r10
 2cf:	4c 89 7d 98          	mov    QWORD PTR [rbp-0x68],r15
 2d3:	4f 8d 14 24          	lea    r10,[r12+r12*1]
 2d7:	4d 89 e3             	mov    r11,r12
 2da:	4d 89 c8             	mov    r8,r9
 2dd:	45 31 ed             	xor    r13d,r13d
 2e0:	48 89 45 b8          	mov    QWORD PTR [rbp-0x48],rax
 2e4:	31 c0                	xor    eax,eax
 2e6:	49 89 c7             	mov    r15,rax
 2e9:	0f 1f 80 00 00 00 00 	nop    DWORD PTR [rax+0x0]
 2f0:	48 8b 7d b8          	mov    rdi,QWORD PTR [rbp-0x48]
 2f4:	49 0f af 38          	imul   rdi,QWORD PTR [r8]
 2f8:	31 f6                	xor    esi,esi
 2fa:	31 c9                	xor    ecx,ecx
 2fc:	31 db                	xor    ebx,ebx
 2fe:	66 90                	xchg   ax,ax
 300:	48 89 f8             	mov    rax,rdi
 303:	49 f7 64 f6 10       	mul    QWORD PTR [r14+rsi*8+0x10]
 308:	49 03 04 f0          	add    rax,QWORD PTR [r8+rsi*8]
 30c:	48 83 d2 00          	adc    rdx,0x0
 310:	48 01 c1             	add    rcx,rax
 313:	48 11 d3             	adc    rbx,rdx
 316:	49 89 0c f0          	mov    QWORD PTR [r8+rsi*8],rcx
 31a:	48 83 c6 01          	add    rsi,0x1
 31e:	48 89 d9             	mov    rcx,rbx
 321:	31 db                	xor    ebx,ebx
 323:	49 39 f4             	cmp    r12,rsi
 326:	75 d8                	jne    300 <crypto_montgomery_mul_ct+0x1c0>
 328:	4d 39 d3             	cmp    r11,r10
 32b:	73 1d                	jae    34a <crypto_montgomery_mul_ct+0x20a>
 32d:	4c 89 d8             	mov    rax,r11
 330:	49 03 0c c1          	add    rcx,QWORD PTR [r9+rax*8]
 334:	48 83 d3 00          	adc    rbx,0x0
 338:	49 89 0c c1          	mov    QWORD PTR [r9+rax*8],rcx
 33c:	48 83 c0 01          	add    rax,0x1
 340:	48 89 d9             	mov    rcx,rbx
 343:	31 db                	xor    ebx,ebx
 345:	49 39 c2             	cmp    r10,rax
 348:	75 e6                	jne    330 <crypto_montgomery_mul_ct+0x1f0>
 34a:	49 83 c5 01          	add    r13,0x1
 34e:	49 01 cf             	add    r15,rcx
 351:	49 83 c0 08          	add    r8,0x8
 355:	49 83 c3 01          	add    r11,0x1
 359:	4d 39 ec             	cmp    r12,r13
 35c:	75 92                	jne    2f0 <crypto_montgomery_mul_ct+0x1b0>
 35e:	4c 8b 6d a0          	mov    r13,QWORD PTR [rbp-0x60]
 362:	4c 89 7d b8          	mov    QWORD PTR [rbp-0x48],r15
 366:	31 c0                	xor    eax,eax
 368:	31 ff                	xor    edi,edi
 36a:	4c 89 55 a0          	mov    QWORD PTR [rbp-0x60],r10
 36e:	4c 8b 55 b0          	mov    r10,QWORD PTR [rbp-0x50]
 372:	4d 01 cd             	add    r13,r9
 375:	0f 1f 00             	nop    DWORD PTR [rax]
 378:	4d 8b 5c c6 10       	mov    r11,QWORD PTR [r14+rax*8+0x10]
 37d:	48 89 f9             	mov    rcx,rdi
 380:	49 8b 54 c5 00       	mov    rdx,QWORD PTR [r13+rax*8+0x0]
 385:	31 db                	xor    ebx,ebx
 387:	49 89 ff             	mov    r15,rdi
 38a:	4c 01 d9             	add    rcx,r11
 38d:	48 89 d6             	mov    rsi,rdx
 390:	48 83 d3 00          	adc    rbx,0x0
 394:	45 31 c9             	xor    r9d,r9d
 397:	48 39 ca             	cmp    rdx,rcx
 39a:	4c 89 cf             	mov    rdi,r9
 39d:	48 19 df             	sbb    rdi,rbx
 3a0:	40 0f 92 c7          	setb   dil
 3a4:	4c 29 fe             	sub    rsi,r15
 3a7:	4c 29 de             	sub    rsi,r11
 3aa:	40 0f b6 ff          	movzx  edi,dil
 3ae:	49 89 34 c2          	mov    QWORD PTR [r10+rax*8],rsi
 3b2:	48 83 c0 01          	add    rax,0x1
 3b6:	49 39 c4             	cmp    r12,rax
 3b9:	75 bd                	jne    378 <crypto_montgomery_mul_ct+0x238>
 3bb:	4c 89 c8             	mov    rax,r9
 3be:	48 39 ca             	cmp    rdx,rcx
 3c1:	4c 8b 55 a0          	mov    r10,QWORD PTR [rbp-0x60]
 3c5:	4c 8b 7d 98          	mov    r15,QWORD PTR [rbp-0x68]
 3c9:	48 19 d8             	sbb    rax,rbx
 3cc:	48 8b 45 b8          	mov    rax,QWORD PTR [rbp-0x48]
 3d0:	4c 8b 45 a8          	mov    r8,QWORD PTR [rbp-0x58]
 3d4:	40 0f 93 c6          	setae  sil
 3d8:	4c 8b 4d b0          	mov    r9,QWORD PTR [rbp-0x50]
 3dc:	40 0f b6 f6          	movzx  esi,sil
 3e0:	48 09 c6             	or     rsi,rax
 3e3:	31 c0                	xor    eax,eax
 3e5:	48 89 f7             	mov    rdi,rsi
 3e8:	48 83 ee 01          	sub    rsi,0x1
 3ec:	48 f7 df             	neg    rdi
 3ef:	90                   	nop
 3f0:	49 8b 14 c1          	mov    rdx,QWORD PTR [r9+rax*8]
 3f4:	49 8b 4c c5 00       	mov    rcx,QWORD PTR [r13+rax*8+0x0]
 3f9:	48 21 fa             	and    rdx,rdi
 3fc:	48 21 f1             	and    rcx,rsi
 3ff:	48 09 ca             	or     rdx,rcx
 402:	49 89 14 c0          	mov    QWORD PTR [r8+rax*8],rdx
 406:	48 83 c0 01          	add    rax,0x1
 40a:	49 39 c4             	cmp    r12,rax
 40d:	75 e1                	jne    3f0 <crypto_montgomery_mul_ct+0x2b0>
 40f:	31 c0                	xor    eax,eax
 411:	0f 1f 80 00 00 00 00 	nop    DWORD PTR [rax+0x0]
 418:	48 83 c0 01          	add    rax,0x1
 41c:	49 c7 07 00 00 00 00 	mov    QWORD PTR [r15],0x0
 423:	49 83 c7 08          	add    r15,0x8
 427:	4c 39 d0             	cmp    rax,r10
 42a:	72 ec                	jb     418 <crypto_montgomery_mul_ct+0x2d8>
 42c:	48 8b 45 b0          	mov    rax,QWORD PTR [rbp-0x50]
 430:	31 d2                	xor    edx,edx
 432:	66 0f 1f 44 00 00    	nop    WORD PTR [rax+rax*1+0x0]
 438:	48 83 c2 01          	add    rdx,0x1
 43c:	48 c7 00 00 00 00 00 	mov    QWORD PTR [rax],0x0
 443:	48 83 c0 08          	add    rax,0x8
 447:	4c 39 e2             	cmp    rdx,r12
 44a:	72 ec                	jb     438 <crypto_montgomery_mul_ct+0x2f8>
 44c:	e8 00 00 00 00       	call   451 <crypto_montgomery_mul_ct+0x311>	44d: R_X86_64_PLT32	__errno_location-0x4
 451:	c7 00 00 00 00 00    	mov    DWORD PTR [rax],0x0
 457:	31 c0                	xor    eax,eax
 459:	48 8b 55 c8          	mov    rdx,QWORD PTR [rbp-0x38]
 45d:	64 48 2b 14 25 28 00 00 00 	sub    rdx,QWORD PTR fs:0x28
 466:	75 2c                	jne    494 <crypto_montgomery_mul_ct+0x354>
 468:	48 8d 65 d8          	lea    rsp,[rbp-0x28]
 46c:	5b                   	pop    rbx
 46d:	41 5c                	pop    r12
 46f:	41 5d                	pop    r13
 471:	41 5e                	pop    r14
 473:	41 5f                	pop    r15
 475:	5d                   	pop    rbp
 476:	c3                   	ret
 477:	e8 00 00 00 00       	call   47c <crypto_montgomery_mul_ct+0x33c>	478: R_X86_64_PLT32	__errno_location-0x4
 47c:	c7 00 16 00 00 00    	mov    DWORD PTR [rax],0x16
 482:	b8 ff ff ff ff       	mov    eax,0xffffffff
 487:	eb d0                	jmp    459 <crypto_montgomery_mul_ct+0x319>
 489:	48 83 4c 0c f8 00    	or     QWORD PTR [rsp+rcx*1-0x8],0x0
 48f:	e9 c3 fd ff ff       	jmp    257 <crypto_montgomery_mul_ct+0x117>
 494:	e8 00 00 00 00       	call   499 <crypto_montgomery_mul_ct+0x359>	495: R_X86_64_PLT32	__stack_chk_fail-0x4
