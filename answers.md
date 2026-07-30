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