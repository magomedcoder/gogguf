# GGUF - Technical Specification of the Binary Format

[Русская версия](GGUF-FORMAT-ru.md)

GGUF is a binary container for storing ML model parameters, metadata, and tensor weights.

A single file contains the complete dataset needed to load a model: no external configs, vocabularies, or index files are required.

The key idea of the format is **typed key-value metadata**.

Instead of a fixed list of hyperparameters (as in early GGML formats), each field is written as a `key -> type + value` pair.

New fields can be added without changing the binary structure of the file.

**v1 is not supported** - only v2 and v3.

Supported structure versions: **v2**, **v3**.

---

## 1. File Layout

The file is divided into a descriptive part and a data part.

First come all structures needed to **interpret** the weights (header, metadata, tensor descriptors).

Then come the weight bytes themselves. This order allows metadata to be read and a tensor map to be built without loading gigabytes of data into memory.

| #   | Section     | Contents                                                | Description                                                |
|-----|-------------|---------------------------------------------------------|------------------------------------------------------------|
| 1   | Header      | `magic`, `version`, `tensor_count`, `metadata_kv_count` | File identification and counters for read planning         |
| 2   | Metadata    | `metadata_kv_count` key-value pairs                     | Model hyperparameters, tokenizer, chat template            |
| 3   | Tensor Info | `tensor_count` descriptors                              | Tensor catalog: name, shape, type, offset in data          |
| 4   | Padding     | `0x00` up to a multiple of `ALIGNMENT`                  | Aligns the start of the weight zone for `mmap`             |
| 5   | Tensor Data | raw weight bytes                                        | Main bulk of the file; quantized or full-precision weights |

```
offset 0
-----------------
- Header        -  fixed prefix: identification and counters
-----------------
- Metadata[]    -  hyperparameters, tokenizer, chat template
-----------------
- TensorInfo[]  -  weight catalog: name, size, type, offset
-----------------
- 0x00 padding  -  <- align_up(end_of_tensor_info)
-----------------
- Tensor Data   -  <- tensor_data_start; main bulk of the file
-----------------
```

### Alignment

The data section is aligned for `mmap`: when mapping a file into memory, the OS requires the start of large blocks to be a multiple of the page size or internal alignment.

The value is specified in `general.alignment`.

| Symbol              | Type     | Description                                                    |
|---------------------|----------|----------------------------------------------------------------|
| `ALIGNMENT`         | `uint32` | Alignment boundary size; default 32, multiple of 8             |
| `align_up(x, a)`    | formula  | Round `x` up to a multiple of `a`                              |
| `tensor_data_start` | `uint64` | Absolute offset of the weight zone start in the file           |
| `file_offset(T)`    | `uint64` | Absolute offset of tensor `T` in the file                      |
| `T.offset`          | `uint64` | Tensor offset from `tensor_data_start`, not from file start    |

```go
// default 32 - value from the spec if general.alignment is absent
const defaultAlignment int64 = 32 // fallback for general.alignment (uint32)

// alignment: metadata["general.alignment"] -> ALIGNMENT, otherwise 32
// Logic: writer and reader must use the same step; otherwise tensor offsets will not match
func alignment(metadata map[string]any) int64 {
    if v, ok := metadata["general.alignment"].(uint32); ok {
        return int64(v) // explicitly set in file, usually 32 for mmap
    }
	
    return defaultAlignment // key not written - use spec default
}

// alignUp(x, a) = x + (a - x%a) % a
// Logic: if x is already a multiple of a, the formula returns x; otherwise pad with zeros to the boundary
func alignUp(x, a int64) int64 {
    return x + (a-x%a)%a
}

// tensor_data_start = align_up(end_of_tensor_info, ALIGNMENT)
// Logic: tensor info is written densely; padding only before the first tensor in the weight zone
func tensorDataStart(endOfTensorInfo int64, align int64) int64 {
    return alignUp(endOfTensorInfo, align)
}

// file_offset(T) = tensor_data_start + T.offset
// Logic: absolute address = weight zone base + relative offset from descriptor
func fileOffset(dataStart int64, t TensorInfo) int64 {
    return dataStart + int64(t.Offset)
}
```

`T.offset` and `tensor_data_start` are multiples of `ALIGNMENT`.

Between tensors - padding up to `ALIGNMENT`, so each tensor starts on an aligned boundary.

Sections 1–3 (header, metadata, tensor info) are written **densely, without padding** - their size is usually small compared to the weights.

---

## 2. Format Versions

The version in the header (`uint32`) changes only on **structural** format changes.

Adding new metadata keys does not increase the version.

| Field               | v1       | v2       | v3        | Description                         |
|---------------------|----------|----------|-----------|-------------------------------------|
| `tensor_count`      | `uint32` | `uint64` | `uint64`  | Number of tensors in the file       |
| `metadata_kv_count` | `uint32` | `uint64` | `uint64`  | Number of metadata pairs            |
| `dimensions[i]`     | `uint32` | `uint64` | `uint64`  | i-th tensor dimension               |
| Endianness          | LE       | LE       | LE or BE  | Byte order of multi-byte fields     |

**v2** - extension to 64 bits for large models.  
**v3** - same layout + big-endian definition (section 3.2).  
**v1** - **not supported**; if `version == 1` - parsing error (see section 15).

---

## 3. Header

The header is a minimal prefix for validation and read planning. `tensor_count` and `metadata_kv_count` are intentionally moved out of metadata: the parser knows how many records to read before parsing KV.

### 3.1. Fields (v2/v3)

| Offset | Size | Field               | Type       | Description                                                          |
|--------|------|---------------------|------------|----------------------------------------------------------------------|
| 0      | 4    | `magic`             | `uint8[4]` | Signature `"GGUF"` (`0x47 0x47 0x55 0x46`); invalid value - not GGUF |
| 4      | 4    | `version`           | `uint32`   | Structure version: **2** or **3**; `1` - not supported               |
| 8      | 8    | `tensor_count`      | `uint64`   | Number of entries in the tensor info array                           |
| 16     | 8    | `metadata_kv_count` | `uint64`   | Number of key-value pairs in metadata                                |

```go
// Header: offset 0..23, section 1
// Logic: counters in header allow pre-allocating memory for metadata and tensor info
type Header struct {
    Magic           [4]byte // offset 0, uint8[4], "GGUF" - first parser check
    Version         uint32 // offset 4; 2|3, v1 reject - determines field width and endianness
    TensorCount     uint64 // offset 8 - how many times to read tensor info entry
    MetadataKVCount uint64 // offset 16 - how many times to read KV; immediately after header
}
```

Byte order - per section 3.2.

### 3.2. Endianness Detection (v3)

By default GGUF is **little-endian**.

v3 adds big-endian support. There is no separate "byte order" field: the order is determined by the last byte of the `version` field.

| Step | Action                  | Description                                    |
|------|-------------------------|------------------------------------------------|
| 1    | `read magic[4]`         | Verify `"GGUF"` signature                      |
| 2    | `seek +3`               | Jump to byte [7] (4th byte of `version` field) |
| 3    | `marker = read uint8`   | Endianness heuristic                           |
| 4    | `marker != 0`           | File is big-endian                             |
| 5    | `seek -4`               | Return to start of `version` field (offset 4)  |
| 6    | `version = read uint32` | Read version with selected byte order          |

| `version` | bytes [4..7]  | marker [7] | byte order | Description            |
|-----------|---------------|------------|------------|------------------------|
| 2         | `02 00 00 00` | `0x00`     | LE         | Standard LE file v2    |
| 3         | `03 00 00 00` | `0x00`     | LE         | Standard LE file v3    |
| 3         | `00 00 00 03` | `0x03`     | BE         | BE file v3             |

v2: always little-endian.

```go
// detectEndianness: v3 heuristic by byte [7] of version field (section 3.2)
// Logic: in LE version=3 gives bytes 03 00 00 00, in BE - 00 00 00 03; distinguish by [7]
// v2 is always LE - marker will be 0x00, BE branch won't trigger
func detectEndianness(r io.ReadSeeker) (binary.ByteOrder, error) {
    // magic[4] already read; cursor at offset 4 - start of version
    if _, err := r.Seek(3, io.SeekCurrent); err != nil { // seek -> offset 7
        return nil, err
    }

    var marker int8
    // read marker as LE uint8: we need raw byte [7], not the full version value
    if err := binary.Read(r, binary.LittleEndian, &marker); err != nil {
        return nil, err
    }

    order := binary.LittleEndian // default per spec
    if marker != 0 { // marker=0x03 => BE v3
        order = binary.BigEndian // non-zero marker - sign of BE version layout
    }

    if _, err := r.Seek(-4, io.SeekCurrent); err != nil { // seek -> offset 4
        return nil, err
    }

    // rewind: version, tensor_count and all KV are read with selected byte order

    return order, nil
}
```

Endianness applies to all fields in sections 1–3.

Inside quant blocks (section 10) scale and quants may be stored in a fixed order (usually LE) - this is a property of `ggml_type`, not the container.

---

## 4. Primitives

### 4.1. Integers and Floats

All multi-byte types are read in the file's byte order (section 3.2).

| Type                           | Size | Description                              |
|--------------------------------|------|------------------------------------------|
| `uint8` / `int8`               | 1    | Unsigned / signed 8-bit integer          |
| `uint16` / `int16`             | 2    | 16-bit integer                           |
| `uint32` / `int32` / `float32` | 4    | 32-bit integer or IEEE 754 float         |
| `uint64` / `int64` / `float64` | 8    | 64-bit integer or IEEE 754 double        |

```go
// readUint32/readUint64 - binary.Read wrappers with file byte order (section 3.2)
// Logic: byte order is uniform for all fields in sections 1-3; type fixes read size
func readUint32(r io.Reader, order binary.ByteOrder) (uint32, error) {
    var v uint32
    err := binary.Read(r, order, &v)
    return v, err
}

func readUint64(r io.Reader, order binary.ByteOrder) (uint64, error) {
    var v uint64
    err := binary.Read(r, order, &v)
    return v, err
}

// readScalar reads payload of scalar value_type (not ARRAY, not nested STRING in ARRAY)
// Logic: switch on ValueType - each case reads exactly sizeof(T) bytes from stream
func readScalar(vtype ValueType, r io.Reader, order binary.ByteOrder) (any, error) {
    switch vtype {
    case Uint8:
        var v uint8
        err := binary.Read(r, order, &v)
        return v, err
    case Int8:
        var v int8
        err := binary.Read(r, order, &v)
        return v, err
    case Uint32:
        return readUint32(r, order)
    case Int32:
        var v int32
        err := binary.Read(r, order, &v)
        return v, err
    case Float32:
        var v float32
        err := binary.Read(r, order, &v)
        return v, err
    case Bool:
        var v uint8 // bool in file is 0 or 1, not Go bool
        err := binary.Read(r, order, &v)
        return v == 1, err
    case String:
        return readString(r, order) // separate gguf_string
    case Uint64:
        return readUint64(r, order)
    // ... remaining types 2,3,10,11,12 similarly
    default:
        return nil, fmt.Errorf("unknown value type: %d", vtype)
    }
}
```

### 4.2. String (`gguf_string`)

Strings are **not null-terminated** - length is specified explicitly.

| Field  | Type     | Description                              |
|--------|----------|------------------------------------------|
| `len`  | `uint64` | String length in bytes (not characters)  |
| `data` | `[]byte` | String body in UTF-8, no trailing `\0`   |

```go
// gguf_string: uint64 len + len bytes UTF-8, not null-terminated
// Logic: first learn size, then read exactly len bytes - no \0 search
type String struct {
    Len  uint64 // payload length in bytes (not Unicode runes)
    Data []byte // UTF-8 data[len]
}

// readString decodes one gguf_string from stream
func readString(r io.Reader, order binary.ByteOrder) (string, error) {
    length, err := readUint64(r, order) // len field - always uint64 in v2/v3
    if err != nil {
        return "", err
    }

    data := make([]byte, length) // exact size: don't read extra bytes
    if _, err := io.ReadFull(r, data); err != nil {
        return "", err // EOF before len - file truncated or len corrupted
    }

    return string(data), nil // Go string over UTF-8 without \0 semantics copy
}
```

Metadata key constraints: ASCII, `lower_snake_case` segments separated by `.`, `len` <= 65535.  
Tensor names: `len` <= 64. Examples: `token_embd.weight`, `blk.0.attn_q.weight`.

---

## 5. Metadata KV

Metadata is a self-describing dictionary: each value is preceded by `value_type`.

### 5.1. Entry

| Field        | Type          | Description                                    |
|--------------|---------------|------------------------------------------------|
| `key`        | `gguf_string` | Unique hierarchical field identifier           |
| `value_type` | `uint32`      | Value type (enum section 5.2)                  |
| `value`      | by type       | Scalar or array                                |

Repeated `metadata_kv_count` times, immediately after header.

```go
// metadata KV entry: key + value_type + value
// Logic: value_type tells parser how many and what kind of bytes to read after key
// fixed order: key first, then type, then payload - no length for entire entry
type MetadataKV struct {
    Key       string // gguf_string key - hierarchy via '.', e.g. qwen3.block_count
    ValueType ValueType // uint32 gguf_metadata_value_type - payload discriminator
    Value     any // scalar | []any (ARRAY) - already parsed value
}
```

### 5.2. `ValueType`

| ID  | Name      | Payload       | Description                            |
|-----|-----------|---------------|----------------------------------------|
| 0   | `UINT8`   | `uint8`       | 8-bit unsigned integer                 |
| 1   | `INT8`    | `int8`        | 8-bit signed integer                   |
| 2   | `UINT16`  | `uint16`      | 16-bit unsigned                        |
| 3   | `INT16`   | `int16`       | 16-bit signed                          |
| 4   | `UINT32`  | `uint32`      | 32-bit unsigned                        |
| 5   | `INT32`   | `int32`       | 32-bit signed                          |
| 6   | `FLOAT32` | `float32`     | 32-bit float                           |
| 7   | `BOOL`    | `uint8`       | Logical: `0` = false, `1` = true       |
| 8   | `STRING`  | `gguf_string` | Text string                            |
| 9   | `ARRAY`   | section 5.3   | Homogeneous array of elements          |
| 10  | `UINT64`  | `uint64`      | 64-bit unsigned                        |
| 11  | `INT64`   | `int64`       | 64-bit signed                          |
| 12  | `FLOAT64` | `float64`     | 64-bit float                           |

```go
// gguf_metadata_value_type, ID 0..12
// Logic: self-describing format - value type stored in file, not in parser schema
// new metadata keys can be added without changing structure version
type ValueType uint32

const (
    Uint8 ValueType = iota // 0, payload: uint8
    Int8 // 1, payload: int8
    Uint16 // 2, payload: uint16
    Int16 // 3, payload: int16
    Uint32 // 4, payload: uint32 - often block_count, head_count
    Int32 // 5, payload: int32 - tokenizer.ggml.token_type[]
    Float32 // 6, payload: float32 - epsilon, rope.freq_base
    Bool // 7, payload: uint8 {0,1} - not native Go bool in file
    String // 8, payload: gguf_string - general.architecture etc.
    Array // 9, payload: element_type+length+data - tokens, merges
    Uint64 // 10, payload: uint64 - large dimensions
    Int64 // 11, payload: int64
    Float64 // 12, payload: float64
)
```

### 5.3. Array (`ARRAY`)

| Field          | Type     | Description                                   |
|----------------|----------|-----------------------------------------------|
| `element_type` | `uint32` | Type of each element (0..12, not ARRAY)       |
| `length`       | `uint64` | Number of elements (not bytes)                |
| `payload`      | `[]T`    | Sequence of `element_type` values             |

```go
// ARRAY payload: element_type(uint32) + length(uint64) + elements[]
// Logic: array nested in value - element type first, then count, then data
type Array struct {
    ElementType ValueType // 0..12, != ARRAY - type of one payload element
    Length      uint64 // element count - not bytes, but number of values
    Elements    []any // unpacked elements; strings are separate gguf_string
}

// readMetaValue reads one value field after value_type in KV entry
func readMetaValue(r io.Reader, order binary.ByteOrder) (any, error) {
    vtype, err := readUint32(r, order) // value_type - first value field
    if err != nil {
        return nil, err
    }

    if ValueType(vtype) != Array {
        return readScalar(ValueType(vtype), r, order) // scalar: fixed payload size
    }

    // ARRAY branch: value structure is wider - need element_type and length
    elemType, err := readUint32(r, order) // ARRAY.element_type
    if err != nil {
        return nil, err
    }

    length, err := readUint64(r, order) // ARRAY.length - how many elements follow
    if err != nil {
        return nil, err
    }

    out := make([]any, length) // prealloc by length from file
    for i := uint64(0); i < length; i++ {
        if ValueType(elemType) == String {
            // tokenizer.ggml.tokens: each element is its own len+data, not one blob
            out[i], err = readString(r, order)
        } else {
            out[i], err = readScalar(ValueType(elemType), r, order)
        }
        if err != nil {
            return nil, err // stop on first corrupted entry
        }
    }

    return out, nil // []any - unified container for any metadata type
}
```

For `STRING`: each element is a separate `gguf_string`.

Nested arrays: `element_type = ARRAY`.

### 5.4. Key Schema

| Prefix                   | Description                                          |
|--------------------------|------------------------------------------------------|
| `general.<field>`        | File and model properties as a whole                 |
| `<architecture>.<field>` | Architecture hyperparameters (`qwen3`, `llama`, ...) |
| `tokenizer.<field>`      | Vocabulary, BPE merges, chat template                |
| `<community>.<field>`    | Arbitrary extensions with namespace                  |

`<architecture>` = value of `general.architecture`. One file - one architecture.

---

## 6. Standard Metadata Keys

### 6.1. Required

| Key                            | Type   | Description                                                       |
|--------------------------------|--------|-------------------------------------------------------------------|
| `general.architecture`         | string | Architecture name; defines `<arch>.*` prefix and loader           |
| `general.quantization_version` | uint32 | Quantization scheme version; required for quantized tensors       |
| `general.alignment`            | uint32 | Data alignment; default **32**; multiple of 8                     |

### 6.2. `general.*`

| Key                    | Type     | Description                              |
|------------------------|----------|------------------------------------------|
| `general.name`         | string   | Human-readable model name                |
| `general.author`       | string   | Author or team                           |
| `general.version`      | string   | Model version (e.g. `v1.0`)              |
| `general.organization` | string   | Developing organization                  |
| `general.basename`     | string   | Base architecture name without fine-tune |
| `general.finetune`     | string   | Fine-tuning type: Chat, Instruct, ...    |
| `general.description`  | string   | Text description of the model            |
| `general.size_label`   | string   | Size class: `8B`, `0.6B`, `8x7B`         |
| `general.file_type`    | uint32   | Dominant quantization type (section 7)   |
| `general.license`      | string   | License in SPDX format                   |
| `general.url`          | string   | Model page URL                           |
| `general.tags`         | string[] | Tags for search and cataloging           |
| `general.languages`    | string[] | Supported languages (ISO 639)            |

### 6.3. `<arch>.*`

Transformer hyperparameters.

Numbers may be `uint32` or `uint64`.

| Key                                       | Type            | Description                                   |
|-------------------------------------------|-----------------|-----------------------------------------------|
| `<arch>.context_length`                   | uint32 \ uint64 | Maximum number of tokens in context           |
| `<arch>.embedding_length`                 | uint32 \ uint64 | Embedding dimension (hidden size)             |
| `<arch>.block_count`                      | uint32 \ uint64 | Number of transformer blocks (layers)         |
| `<arch>.feed_forward_length`              | uint32 \ uint64 | Intermediate FFN layer dimension              |
| `<arch>.attention.head_count`             | uint32 \ uint64 | Number of query heads in attention            |
| `<arch>.attention.head_count_kv`          | uint32 \ uint64 | Number of key/value heads (GQA/MQA)           |
| `<arch>.attention.key_length`             | uint32 \ uint64 | K/V dimension per head (head_dim)             |
| `<arch>.attention.layer_norm_rms_epsilon` | float32         | Epsilon in RMSNorm                            |
| `<arch>.rope.freq_base`                   | float32         | Base frequency of Rotary Position Embedding   |
| `<arch>.rope.freq_scale`                  | float32         | RoPE scale factor (long context)              |

### 6.4. `tokenizer.*`

| Key                               | Type     | Description                                       |
|-----------------------------------|----------|---------------------------------------------------|
| `tokenizer.ggml.model`            | string   | Tokenizer family: `gpt2`, `llama`, ...            |
| `tokenizer.ggml.pre`              | string   | Pretokenizer: `default`, `qwen2`, `llama3`, ...   |
| `tokenizer.ggml.tokens`           | string[] | Full vocabulary: index -> token string            |
| `tokenizer.ggml.merges`           | string[] | BPE merge rules: `"token_a token_b"`              |
| `tokenizer.ggml.token_type`       | int32[]  | Type of each token: normal, control, user-defined |
| `tokenizer.ggml.bos_token_id`     | uint32   | Beginning-of-sequence token ID                    |
| `tokenizer.ggml.eos_token_id`     | uint32   | End-of-sequence / stop token ID                   |
| `tokenizer.ggml.padding_token_id` | uint32   | Padding token ID                                  |
| `tokenizer.ggml.add_bos_token`    | bool     | Auto-add BOS on encode                            |
| `tokenizer.ggml.add_eos_token`    | bool     | Auto-add EOS on encode                            |
| `tokenizer.chat_template`         | string   | Jinja2 template for dialog formatting             |

---

## 7. `general.file_type`

Summary indicator of the file's dominant quantization.

| ID  | Name                   | Description                                     |
|-----|------------------------|-------------------------------------------------|
| 0   | `ALL_F32`              | All tensors in FP32                             |
| 1   | `MOSTLY_F16`           | Most tensors FP16                               |
| 2   | `MOSTLY_Q4_0`          | Most tensors Q4_0 (4-bit, symmetric)            |
| 3   | `MOSTLY_Q4_1`          | Most tensors Q4_1 (4-bit, with min)             |
| 4   | `MOSTLY_Q4_1_SOME_F16` | Q4_1 + separate FP16 tensors                    |
| 5   | `MOSTLY_Q4_2`          | Deprecated; removed from llama.cpp              |
| 6   | `MOSTLY_Q4_3`          | Deprecated; removed from llama.cpp              |
| 7   | `MOSTLY_Q8_0`          | Most tensors Q8_0 (8-bit)                       |
| 8   | `MOSTLY_Q5_0`          | Most tensors Q5_0 (5-bit)                       |
| 9   | `MOSTLY_Q5_1`          | Most tensors Q5_1                               |
| 10  | `MOSTLY_Q2_K`          | Most tensors Q2_K (K-quant, 2-bit)              |
| 11  | `MOSTLY_Q3_K_S`        | Q3_K small                                      |
| 12  | `MOSTLY_Q3_K_M`        | Q3_K medium                                     |
| 13  | `MOSTLY_Q3_K_L`        | Q3_K large                                      |
| 14  | `MOSTLY_Q4_K_S`        | Q4_K small                                      |
| 15  | `MOSTLY_Q4_K_M`        | Q4_K medium                                     |
| 16  | `MOSTLY_Q5_K_S`        | Q5_K small                                      |
| 17  | `MOSTLY_Q5_K_M`        | Q5_K medium                                     |
| 18  | `MOSTLY_Q6_K`          | Most tensors Q6_K                               |

---

## 8. Tensor Info

Tensor Info is the weight catalog.

Each entry describes one tensor; size in bytes is computed, not stored.

Repeated `tensor_count` times, after metadata.

| Field          | Type          | Description                                              |
|----------------|---------------|----------------------------------------------------------|
| `name`         | `gguf_string` | Unique tensor name (`blk.0.attn_q.weight`, ...)          |
| `n_dimensions` | `uint32`      | Tensor rank (number of dimensions)                       |
| `dimensions`   | `uint64[n]`   | Size along each axis; product = number of elements       |
| `type`         | `uint32`      | Storage type (`ggml_type`, section 9)                    |
| `offset`       | `uint64`      | Offset from `tensor_data_start`; multiple of `ALIGNMENT` |

| Field           | Constraint         | Description                              |
|-----------------|--------------------|------------------------------------------|
| `n_dimensions`  | >= 1               | Tensor rank; spec allows <= 4            |
| `dimensions[i]` | > 0                | Each axis has positive size              |
| `offset`        | `% ALIGNMENT == 0` | Tensor start on aligned boundary         |

```go
// tensor info entry (section 3): name + n_dimensions + dimensions[] + type + offset
// Logic: descriptor does not contain weights - only how to find and interpret them in tensor data; field order in file is strict; tensor size in bytes is not stored here
type TensorInfo struct {
    Name       string // gguf_string name - unique identifier, e.g. blk.0.attn_q.weight
    Dimensions []uint64 // uint32 n_dimensions + uint64[n] - product = number of elements
    Type       GGMLType // uint32 ggml_type - selects decoder and tensorSize formula
    Offset     uint64 // from tensor_data_start, % ALIGNMENT == 0
}

// size_bytes = ceil(prod(dimensions) / block_values) * block_bytes
// Logic: size not written in file - computed from element count and quant block parameters
func tensorSize(typ GGMLType, dims []uint64) int64 {
    n := uint64(1)
    for _, d := range dims {
        n *= d // prod(dimensions) - total number of tensor weights
    }

    bv, bb := blockParams(typ) // block_values, block_bytes - depend only on ggml_type
    blocks := (n + bv - 1) / bv // ceil(n / bv) - trailing partial block is still whole

    return int64(blocks * bb) // result in bytes for SectionReader and boundary validation
}
```

GGML convention for 2D matrices: `dimensions = [cols, rows]` (input * output).

---

## 9. `ggml_type`

Type defines interpretation of raw bytes.

Quantized types store weights in blocks with a shared scale.

| ID  | Name      | `block_values` | `block_bytes` | Description                                 |
|-----|-----------|----------------|---------------|---------------------------------------------|
| 0   | `F32`     | 1              | 4             | Full precision IEEE 754                     |
| 1   | `F16`     | 1              | 2             | Half precision                              |
| 2   | `Q4_0`    | 32             | 18            | 4-bit symmetric, 32 weights/block           |
| 3   | `Q4_1`    | 32             | 20            | 4-bit with min, 32 weights/block            |
| -   | `Q4_2`    | -              | -             | Removed (id 4)                              |
| -   | `Q4_3`    | -              | -             | Removed (id 5)                              |
| 6   | `Q5_0`    | 32             | 22            | 5-bit symmetric                             |
| 7   | `Q5_1`    | 32             | 26            | 5-bit with min                              |
| 8   | `Q8_0`    | 32             | 34            | 8-bit symmetric; ~4x compression from F32   |
| 9   | `Q8_1`    | 32             | 36            | 8-bit + sum field for dot-product           |
| 10  | `Q2_K`    | 256            | 84            | K-quant 2-bit, block 256                    |
| 11  | `Q3_K`    | 256            | 110           | K-quant 3-bit                               |
| 12  | `Q4_K`    | 256            | 144           | K-quant 4-bit                               |
| 13  | `Q5_K`    | 256            | 176           | K-quant 5-bit                               |
| 14  | `Q6_K`    | 256            | 210           | K-quant 6-bit                               |
| 15  | `Q8_K`    | 256            | 292           | K-quant 8-bit                               |
| 16  | `IQ2_XXS` | -              | var           | Importance quant 2-bit extra-extra-small    |
| 17  | `IQ2_XS`  | -              | var           | Importance quant 2-bit extra-small          |
| 18  | `IQ3_XXS` | -              | var           | Importance quant 3-bit extra-extra-small    |
| 19  | `IQ1_S`   | -              | var           | Importance quant 1-bit small                |
| 20  | `IQ4_NL`  | -              | var           | Importance quant 4-bit non-linear           |
| 21  | `IQ3_S`   | -              | var           | Importance quant 3-bit small                |
| 22  | `IQ2_S`   | -              | var           | Importance quant 2-bit small                |
| 23  | `IQ4_XS`  | -              | var           | Importance quant 4-bit extra-small          |
| 24  | `I8`      | 1              | 1             | 8-bit integer                               |
| 25  | `I16`     | 1              | 2             | 16-bit integer                              |
| 26  | `I32`     | 1              | 4             | 32-bit integer                              |
| 27  | `I64`     | 1              | 8             | 64-bit integer                              |
| 28  | `F64`     | 1              | 8             | Double precision                            |
| 29  | `IQ1_M`   | -              | var           | Importance quant 1-bit medium               |
| 30  | `BF16`    | 1              | 2             | Brain float 16                              |
| 34  | `TQ1_0`   | -              | var           | Ternary quant 1.0                           |
| 35  | `TQ2_0`   | -              | var           | Ternary quant 2.0                           |
| 39  | `MXFP4`   | -              | var           | Microscaling FP4                            |

```go
// ggml_type (uint32), defines tensor data layout
// Logic: same byte slice read differently depending on ggml_type; ID in enum matches numeric value in tensor type field
type GGMLType uint32

const (
    F32 GGMLType = iota // 0, block_values=1, block_bytes=4 - no quantization
    F16 // 1, block_values=1, block_bytes=2
    Q4_0 // 2, block_values=32, block_bytes=18 - 4-bit symmetric
    Q4_1 // 3, block_values=32, block_bytes=20 - 4-bit + min
    _ // 4, Q4_2 deprecated - id reserved, skip in enum
    _ // 5, Q4_3 deprecated - otherwise subsequent type IDs shift
    Q5_0 // 6, block_values=32, block_bytes=22
    Q5_1 // 7, block_values=32, block_bytes=26
    Q8_0 // 8, block_values=32, block_bytes=34 - primary type in gguf.go
    Q8_1 // 9, block_values=32, block_bytes=36 - + sum for matmul
    Q2_K // 10, block_values=256, block_bytes=84 - K-quant family
    Q3_K // 11, block_values=256, block_bytes=110
    Q4_K // 12, block_values=256, block_bytes=144
    Q5_K // 13, block_values=256, block_bytes=176
    Q6_K // 14, block_values=256, block_bytes=210
    Q8_K // 15, block_values=256, block_bytes=292
    // ...
)

// blockParams returns (block_values, block_bytes) for ggml_type
// Logic: lookup table from section 9; F32/I8 - 1 value per block, Q*_0 - 32, Q*_K - 256
func blockParams(typ GGMLType) (uint64, uint64) {
    switch typ {
    case F32:
        return 1, 4 // block_values=1, block_bytes=4
    case Q8_0:
        return 32, 34
    case Q4_0:
        return 32, 18
    // ... remaining types from section 9 table
    default:
        return 0, 0
    }
}
```

---

## 10. Quant Block Layout

A quant block is the minimum unit of dequantization.

### 10.1. Q4_0 - 18 bytes, 32 values

| Offset | Size | Field | Type     | Description                       |
|--------|------|-------|----------|-----------------------------------|
| 0      | 2    | `d`   | float16  | Block scale                       |
| 2      | 16   | `qs`  | `[]byte` | 32 * 4-bit, 2 nibbles per byte    |

```go
// Q4_0 block: d(float16) + qs[16]; 18 bytes -> 32 float32
// w[i] = d * (q[i] - 8), zero-point=8
// Logic: one scale d per 32 weights; 4-bit q packed 2 per byte; symmetry around 8
func dequantQ4_0(block []byte) ([32]float32, error) {
    d := fp16ToF32(binary.LittleEndian.Uint16(block[0:2])) // offset 0, field d - shared scale
	
    var out [32]float32 // exactly block_values for Q4_0
    for i := 0; i < 32; i++ {
        // i%2==0: lower 4 bits of byte; i%2==1: upper 4 bits of same byte
        q := (block[2+i/2] >> (4 * (i % 2))) & 0x0F // qs: 2 nibble/byte
        out[i] = d * float32(int(q)-8) // zero-point 8 centers range 0..15
    }
	
    return out, nil // 32 restored float32 for matmul/dequant pipeline
}
```

### 10.2. Q4_1 - 20 bytes, 32 values

| Offset | Size | Field | Type     | Description       |
|--------|------|-------|----------|-------------------|
| 0      | 2    | `d`   | float16  | Scale             |
| 2      | 2    | `m`   | float16  | Offset (min)      |
| 4      | 16   | `qs`  | `[]byte` | 32 * 4-bit        |

```go
// Q4_1 block: d(float16) + m(float16) + qs[16]; 20 bytes -> 32 float32
// w[i] = d * q[i] + m
// Logic: unlike Q4_0, range defined linearly via min (m), no zero-point
func dequantQ4_1(block []byte) ([32]float32, error) {
    d := fp16ToF32(binary.LittleEndian.Uint16(block[0:2])) // offset 0, field d - scale step
    m := fp16ToF32(binary.LittleEndian.Uint16(block[2:4])) // offset 2, field m - min offset

    var out [32]float32
    for i := 0; i < 32; i++ {
        q := (block[4+i/2] >> (4 * (i % 2))) & 0x0F // offset 4, field qs - packing as in Q4_0
        out[i] = d*float32(q) + m // affine: q∈[0,15] stretched, m sets base
    }
	
    return out, nil
}
```

### 10.3. Q5_0 - 22 bytes, 32 values

| Offset | Size | Field | Type      | Description                      |
|--------|------|-------|-----------|----------------------------------|
| 0      | 2    | `d`   | float16   | Scale                            |
| 2      | 4    | `qh`  | uint8[4]  | Upper bits of 5-bit quants       |
| 6      | 16   | `qs`  | uint8[16] | Lower 4 bits of each quant       |

### 10.4. Q8_0 - 34 bytes, 32 values

| Offset | Size | Field | Type     | Description                 |
|--------|------|-------|----------|-----------------------------|
| 0      | 2    | `d`   | float16  | Block scale                 |
| 2      | 32   | `qs`  | `[]int8` | 32 signed 8-bit quants      |

```go
// Q8_0 block: d(float16) + qs[int8;32]; 34 bytes -> 32 float32
// w[i] = d * qs[i]; d usually LE inside block
// Logic: each weight is signed int8; higher precision than Q4, but block almost 2x larger; scale d inside block is ggml_type property, not GGUF container byte order
func dequantQ8_0(block []byte) ([32]float32, error) {
    d := fp16ToF32(binary.LittleEndian.Uint16(block[0:2])) // offset 0, field d
    var out [32]float32
    for i := 0; i < 32; i++ {
        out[i] = d * float32(int8(block[2+i])) // qs[i] - separate byte, no nibble packing
    }

    return out, nil
}
```

### 10.5. Q8_1 - 36 bytes, 32 values

| Offset | Size | Field | Type     | Description                                   |
|--------|------|-------|----------|-----------------------------------------------|
| 0      | 2    | `d`   | float16  | Scale                                         |
| 2      | 2    | `s`   | float16  | Precomputed sum of quants (for matmul)        |
| 4      | 32   | `qs`  | int8[32] | 32 signed quants                              |

### 10.6. K-quants (Q2_K ... Q8_K)

`block_values = 256`. Inside: sub-block scales, bit masks, packed nibbles. `block_bytes` is fixed (section 9).

### 10.7. F32 / F16 / I8 / I16 / I32

Direct storage without dequantization: one element = one block.

---

## 11. Tensor Data

| Element          | Description                                           |
|------------------|-------------------------------------------------------|
| Padding          | Zero bytes `0x00` from end of tensor info to `align_up` |
| `tensor_data`    | Continuous zone of raw bytes of all tensors           |
| `T.offset`       | Tensor start inside `tensor_data`                       |
| Inter-tensor gap | Padding `0x00` to next aligned `offset`               |

Data may differ from the source model due to quantization or transposition - interpretation is defined by tensor info + metadata.

```go
// padding 0x00: seek from pos to align_up(pos, align)
// Logic: writer appends zeros to boundary; reader skips them, not parsing as data; pos - end of tensor info; dataStart - absolute offset of tensor data start in file
func skipPadding(r io.ReadSeeker, pos, align int64) (int64, error) {
    dataStart := alignUp(pos, align) // tensor_data_start
    if _, err := r.Seek(dataStart, io.SeekStart); err != nil {
        return 0, err // seek beyond file - pos/align invalid
    }

    return dataStart, nil // return computed offset for TensorOffset
}
```

---

## 12. Parsing Algorithm

Two stages: (1) descriptive part, (2) lazy access to weights.

| Step | Action                  | Description                            |
|------|-------------------------|----------------------------------------|
| 1    | Verify `magic`          | Format validation                      |
| 2    | Determine byte order    | section 3.2                            |
| 3    | Read `version`          | Accept 2 or 3                          |
| 4    | Read counters           | `tensor_count`, `metadata_kv_count`    |
| 5    | Parse metadata          | `metadata_kv_count` KV pairs           |
| 6    | Get `alignment`         | From metadata, default 32              |
| 7    | Parse tensor info       | `tensor_count` descriptors             |
| 8    | Compute `data_start`    | `align_up` of current position         |
| 9    | Read weights on demand  | `slice(data_start + offset, size)`     |

```go
// Parse: sections 1-3 (header, metadata, tensor info); tensor data - lazy
// Logic: two-phase parsing - first entire file "map", weights read on demand; after Parse cursor at tensor_data_start, but weight bytes not copied in File
func Parse(r io.ReadSeeker) (*File, error) {
    // --- step 1: magic ---
    var magic [4]byte
    if _, err := io.ReadFull(r, magic[:]); err != nil {
        return nil, err
    }

    if !bytes.Equal(magic[:], []byte("GGUF")) {
        return nil, fmt.Errorf("not a GGUF file") // early exit: don't waste time parsing
    }

    // step 2: byte order (before reading version and all uint64)
    byteOrder, err := detectEndianness(r) // v3: LE|BE; v2: always LE
    if err != nil {
        return nil, err
    }

    // step 3: version
    version, err := readUint32(r, byteOrder)
    if err != nil {
        return nil, err
    }
    if version != 2 && version != 3 {
        return nil, fmt.Errorf("unsupported version: %d (v1 not supported)", version)
    }

    // step 4: counters from header (already after magic+version in stream)
    tensorCount, err := readUint64(r, byteOrder) // how many tensor info entries
    if err != nil {
        return nil, err
    }

    metadataCount, err := readUint64(r, byteOrder) // how many KV pairs in metadata
    if err != nil {
        return nil, err
    }

    // step 5: metadata KV (section 2)
    metadata := make(map[string]any, metadataCount)
    for i := uint64(0); i < metadataCount; i++ {
        key, err := readString(r, byteOrder) // KV.key
        if err != nil {
            return nil, err
        }

        val, err := readMetaValue(r, byteOrder) // KV.value_type + KV.value
        if err != nil {
            return nil, err
        }

        metadata[key] = val // map gives O(1) access to general.* and <arch>.*
    }

    // step 6: ALIGNMENT from metadata
    alignment := int64(32)
    if v, ok := metadata["general.alignment"].(uint32); ok {
        alignment = int64(v) // otherwise padding and tensor offsets won't match writer
    }

    // step 7: tensor info (section 3)
    tensors := make([]TensorInfo, tensorCount)
    for i := uint64(0); i < tensorCount; i++ {
        name, err := readString(r, byteOrder) // name
        if err != nil {
            return nil, err
        }

        ndim, err := readUint32(r, byteOrder) // n_dimensions - tensor rank
        if err != nil {
            return nil, err
        }

        dims := make([]uint64, ndim)
        for j := uint32(0); j < ndim; j++ {
            dims[j], err = readUint64(r, byteOrder) // dimensions[j] - axis j size
            if err != nil {
                return nil, err
            }
        }

        typ, err := readUint32(r, byteOrder) // type (ggml_type)
        if err != nil {
            return nil, err
        }

        offset, err := readUint64(r, byteOrder) // offset - relative to tensor data
        if err != nil {
            return nil, err
        }

        tensors[i] = TensorInfo{
            Name:       name,
            Dimensions: dims,
            Type:       GGMLType(typ),
            Offset:     offset,
        }
    }

    // step 8: tensor_data_start
    pos, err := r.Seek(0, io.SeekCurrent) // end of tensor info - start of padding
    if err != nil {
        return nil, err
    }
    dataStart := alignUp(pos, alignment) // tensor_data_start; beyond - only weights

    return &File{
        Version:      int(version),
        ByteOrder:    byteOrder,
        Metadata:     metadata,
        Tensors:      tensors,
        TensorOffset: dataStart, // anchor for all TensorData()
    }, nil
}

// slice [tensor_data_start+offset, tensor_data_start+offset+size)
// Logic: SectionReader doesn't copy gigabytes to RAM - read window of needed tensor; ReaderAt allows mmap file read by offset without sequential seek
func (f *File) TensorData(r io.ReaderAt, t TensorInfo) (io.Reader, error) {
    start := f.TensorOffset + int64(t.Offset) // absolute offset in file
    size := tensorSize(t.Type, t.Dimensions) // slice boundary - only this tensor
    return io.NewSectionReader(r, start, size), nil
}
```

---

## 13. Complete Structure (Go)

| Structure / field        | Description                                 |
|--------------------------|---------------------------------------------|
| `File`                   | Root layout of entire file                  |
| `Header.Magic`           | `"GGUF"` signature                          |
| `Header.Version`         | Format version: 2 or 3                      |
| `Header.TensorCount`     | Number of tensors                           |
| `Header.MetadataKVCount` | Number of KV pairs                          |
| `Metadata`               | `map[string]any` of metadata                |
| `Tensors`                | Slice of tensor descriptors                 |
| `TensorOffset`           | Absolute offset of `tensor_data` start      |
| `MetadataKV.Key`         | Metadata field name                         |
| `MetadataKV.ValueType`   | Value type                                  |
| `MetadataKV.Value`       | Scalar or array                             |
| `TensorInfo.Name`        | Tensor name                                 |
| `TensorInfo.Dimensions`  | Sizes along axes                            |
| `TensorInfo.Type`        | `GGMLType`                                  |
| `TensorInfo.Offset`      | Offset in `tensor_data`                     |
| `String.Len`             | String length in bytes                      |
| `String.Data`            | UTF-8 string body                           |

```go
// Logic: File is Parse() result; links metadata, tensor catalog and weight entry point; without TensorOffset cannot compute absolute address of any tensor
type File struct {
    Version      int // header.version - affects field width and endianness
    ByteOrder    binary.ByteOrder // LE|BE, sections 1-3; quant blocks may be LE inside
    Metadata     map[string]any // metadata KV map - hyperparams, tokenizer, general.*
    Tensors      []TensorInfo // tensor info array - catalog without loading weights
    TensorOffset int64 // align_up(end_of_tensor_info) - base for all offsets
}

// Header - wire format of first 24 bytes; in Parse read field by field, not one struct Read
type Header struct {
    Magic           [4]byte // uint8[4] - signature for validation
    Version         uint32 // after magic; determines reject v1 and detectEndianness
    TensorCount     uint64 // sets length of tensor info read loop
    MetadataKVCount uint64 // sets length of metadata read loop
}

// MetadataKV - logical unit of section 2; in map only Key -> Value is stored
type MetadataKV struct {
    Key       string // gguf_string - unique field identifier
    ValueType ValueType // uint32 - payload discriminator; without it cannot read value
    Value     any // parsed value - scalar or slice for ARRAY
}

// TensorInfo - logical unit of section 3; weights lie separately in tensor data
type TensorInfo struct {
    Name       string // gguf_string - key for weight lookup by layer name
    Dimensions []uint64 // uint64[n_dimensions] - shape; needed for tensorSize
    Type       GGMLType // uint32 - byte decoder choice (F32, Q8_0, ...)
    Offset     uint64 // % ALIGNMENT == 0 - offset from TensorOffset
}

// String - helper representation of gguf_string when reading from stream
type String struct {
    Len  uint64 // uint64 - payload length in bytes
    Data []byte // UTF-8[len], no \0 - exactly len bytes from file
}
```

---

## 14. Invariants and Errors

| Condition                             | Action     | Description                         |
|---------------------------------------|------------|-------------------------------------|
| `magic != "GGUF"`                     | reject     | File is not GGUF                    |
| `version == 1`                        | reject     | v1 not supported                    |
| `version not in {2, 3}`               | reject     | Unsupported structure version       |
| `BOOL not in {0, 1}`                  | reject     | Invalid logical value               |
| `value_type` unknown                  | reject     | Parser cannot read value            |
| `T.offset % ALIGNMENT != 0`           | reject     | Tensor alignment violated           |
| `general.alignment % 8 != 0`          | reject     | Invalid alignment value             |
| missing `general.architecture`        | incomplete | Model architecture unknown          |
| quants without `quantization_version` | incomplete | Quantization scheme version unknown |

The reader must accept both `uint32` and `uint64` for numeric metadata fields.

---

## 15. v1 (not supported)

v1 format is **not read** by modern parsers. A file with `version == 1` is rejected before parsing metadata and tensor info.

v1 vs v2 differences (reference, for identifying obsolete files):

| Field               | v1       | Description                   |
|---------------------|----------|-------------------------------|
| `tensor_count`      | `uint32` | 32-bit tensor counter         |
| `metadata_kv_count` | `uint32` | 32-bit KV counter             |
| `dimensions[i]`     | `uint32` | 32-bit dimensions             |
| endianness          | LE only  | Little-endian only            |
