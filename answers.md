# 2

## 2.1
 chr(0)

'\x00'

 print(chr(0))


 "this is a test" + chr(0) + "string"

'this is a test\x00string'

 print("this is a test" + chr(0) + "string")

this is a teststring

0在print时不能显示

## 2.2

(a)
using fewer bytes, leading to a better compression ratio and shorter token sequences

(b) 
decode_utf8_bytes_to_str_wrong("你".encode("utf-8"))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 2, in decode_utf8_bytes_to_str_wrong
  File "<stdin>", line 2, in <genexpr>
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe4 in position 0: unexpected end of data

"One byte does not necessarily correspond to one Unicode character!" -- > UTF-8 characters may consist of multiple bytes, which must be decoded together rather than one byte at a time.

b"\xff\xff".decode("utf-8")
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 0: invalid start byte


## 2.5

### tinystorys

(a) reasonable

Training time: 227.82 seconds
Vocabulary size: 10000
Number of merges: 9743
Longest token id: 7342
Longest token length: 15 bytes
Longest token bytes: b' accomplishment'
Maximum resident set size (kbytes): 1270992

1000 - (256 + 1) = 9743

(b)

耗时来源：

1.BPE merge：每轮都要遍历candidate_pair，并调用max找到频率最高（双标准）的 pair
2.将所有单词转化为单byte tumple


### openweb

(a) 
Training time: 28407.27 seconds 
Vocabulary size: 32000 
Number of merges: 31743 
Longest token id: 25822 
Longest token length: 64 bytes 
Longest token decoded: ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ

reasonable:noisy or incorrectly text exists.

compare（抽取了十篇文档）:

TinyStories tokenizer on TinyStories :     7775 bytes,     1930 tokens, 4.0285 bytes/token
OWT tokenizer        on TinyStories :     7775 bytes,     2003 tokens, 3.8817 bytes/token
TinyStories tokenizer on OpenWebText :    50180 bytes,    16206 tokens, 3.0964 bytes/token
OWT tokenizer        on OpenWebText :    50180 bytes,    11877 tokens, 4.2250 bytes/token

在tinystory上 owttokenizer有轻微劣势 但在owt上有明显优势 说明其泛化能力更强 

eg：

TinyStories tokenizer (35 tokens):
[b'Three', b' s', b'en', b'i', b'or', b' adm', b'in', b'ist', b'r', b'ation', b' offic', b'ial', b's', b' told', b' Re', b'ut', b'ers', b' that', b' the', b' pres', b'ident', b' is', b' consid', b'ering', b' putting', b' an', b' import', b' tar', b'iff', b' on', b' Ch', b'ine', b'se', b' steel', b'.']

OpenWebText tokenizer (19 tokens):
[b'Three', b' senior', b' administration', b' officials', b' told', b' Reuters', b' that', b' the', b' president', b' is', b' considering', b' putting', b' an', b' import', b' tariff', b' on', b' Chinese', b' steel', b'.']


uint16：0 65535 token-size:roughly 10000/32000

