# GGUF - техническая спецификация бинарного формата

GGUF - бинарный контейнер для хранения параметров ML-модели, метаданных и весов тензоров.

Один файл содержит полный набор данных для загрузки модели: не нужны внешние конфиги, словари или индексные файлы.

Ключевая идея формата - **типизированные key-value метаданные**.

Вместо фиксированного списка гиперпараметров (как в ранних форматах GGML) каждое поле записывается как пара `ключ -> тип + значение`. 

Новые поля добавляются без изменения бинарной структуры файла.

**v1 не поддерживается** - только v2 и v3.

Поддерживаемые версии структуры: **v2**, **v3**.

---

## 1. Layout файла

Файл разделён на описательную и данныхую части.

Сначала идут все структуры, необходимые для **интерпретации** весов (заголовок, метаданные, дескрипторы тензоров).

Затем - сами байты весов. Такой порядок позволяет прочитать метаданные и построить карту тензоров, не загружая гигабайты данных в память.

| #   | Секция      | Содержимое                                              | Описание                                               |
|-----|-------------|---------------------------------------------------------|--------------------------------------------------------|
| 1   | Header      | `magic`, `version`, `tensor_count`, `metadata_kv_count` | Идентификация файла и счётчики для планирования чтения |
| 2   | Metadata    | `metadata_kv_count` пар key-value                       | Гиперпараметры модели, tokenizer, chat template        |
| 3   | Tensor Info | `tensor_count` дескрипторов                             | Каталог тензоров: имя, форма, тип, смещение в данных   |
| 4   | Padding     | `0x00` до кратности `ALIGNMENT`                         | Выравнивание начала зоны весов для `mmap`              |
| 5   | Tensor Data | сырые байты весов                                       | Основной объём файла; квантизованные или полные веса   |

```
offset 0
-----------------
- Header        -  фиксированный префикс: идентификация и счётчики
-----------------
- Metadata[]    -  гиперпараметры, tokenizer, chat template
-----------------
- TensorInfo[]  -  каталог весов: имя, размер, тип, смещение
-----------------
- 0x00 padding  -  <- align_up(end_of_tensor_info)
-----------------
- Tensor Data   -  <- tensor_data_start; основной объём файла
-----------------
```

### Выравнивание

Секция данных выравнивается для `mmap`: при отображении файла в память ОС требует, чтобы начало больших блоков было кратно размеру страницы или внутреннему alignment.

Значение задаётся в `general.alignment`.

| Символ              | Тип      | Описание                                                    |
|---------------------|----------|-------------------------------------------------------------|
| `ALIGNMENT`         | `uint32` | Размер границы выравнивания; default 32, кратно 8           |
| `align_up(x, a)`    | формула  | Округление `x` вверх до кратного `a`                        |
| `tensor_data_start` | `uint64` | Абсолютное смещение начала зоны весов в файле               |
| `file_offset(T)`    | `uint64` | Абсолютное смещение тензора `T` в файле                     |
| `T.offset`          | `uint64` | Смещение тензора от `tensor_data_start`, не от начала файла |

```go
// default 32 - значение из спецификации, если general.alignment отсутствует
const defaultAlignment int64 = 32 // fallback для general.alignment (uint32)

// alignment: metadata["general.alignment"] -> ALIGNMENT, иначе 32
// Логика: writer и reader должны использовать один шаг; иначе offset тензоров не сойдутся
func alignment(metadata map[string]any) int64 {
    if v, ok := metadata["general.alignment"].(uint32); ok {
        return int64(v) // явно задано в файле, обычно 32 для mmap
    }
	
    return defaultAlignment // ключ не записан - берём дефолт спецификации
}

// alignUp(x, a) = x + (a - x%a) % a
// Логика: если x уже кратно a, формула возвращает x; иначе дополняем нулями до границы
func alignUp(x, a int64) int64 {
    return x + (a-x%a)%a
}

// tensor_data_start = align_up(end_of_tensor_info, ALIGNMENT)
// Логика: tensor info пишется плотно; padding только перед первым тензором в зоне весов
func tensorDataStart(endOfTensorInfo int64, align int64) int64 {
    return alignUp(endOfTensorInfo, align)
}

// file_offset(T) = tensor_data_start + T.offset
// Логика: абсолютный адрес = база зоны весов + относительное смещение из дескриптора
func fileOffset(dataStart int64, t TensorInfo) int64 {
    return dataStart + int64(t.Offset)
}
```

`T.offset` и `tensor_data_start` кратны `ALIGNMENT`.

Между тензорами - padding до `ALIGNMENT`, чтобы каждый тензор начинался на выровненной границе.

Секции 1–3 (header, metadata, tensor info) пишутся **плотно, без padding** - их размер обычно мал по сравнению с весами.

---

## 2. Версии формата

Версия в header (`uint32`) меняется только при **структурных** изменениях формата.

Добавление новых ключей метаданных версию не увеличивает.

| Поле                | v1       | v2       | v3        | Описание                         |
|---------------------|----------|----------|-----------|----------------------------------|
| `tensor_count`      | `uint32` | `uint64` | `uint64`  | Число тензоров в файле           |
| `metadata_kv_count` | `uint32` | `uint64` | `uint64`  | Число пар метаданных             |
| `dimensions[i]`     | `uint32` | `uint64` | `uint64`  | i-я размерность тензора          |
| Endianness          | LE       | LE       | LE или BE | Порядок байт многобайтовых полей |

**v2** - расширение до 64 бит для больших моделей.  
**v3** - тот же layout + определение big-endian (раздел 3.2).  
**v1** - **не поддерживается**; при `version == 1` - ошибка парсинга (см. раздел 15).

---

## 3. Header

Header - минимальный префикс для валидации и планирования чтения. `tensor_count` и `metadata_kv_count` вынесены из метаданных намеренно: парсер знает, сколько записей читать, ещё до разбора KV.

### 3.1. Поля (v2/v3)

| Offset | Size | Field               | Type       | Описание                                                                |
|--------|------|---------------------|------------|-------------------------------------------------------------------------|
| 0      | 4    | `magic`             | `uint8[4]` | Сигнатура `"GGUF"` (`0x47 0x47 0x55 0x46`); неверное значение - не GGUF |
| 4      | 4    | `version`           | `uint32`   | Версия структуры: **2** или **3**; `1` - не поддерживается              |
| 8      | 8    | `tensor_count`      | `uint64`   | Сколько записей в tensor info array                                     |
| 16     | 8    | `metadata_kv_count` | `uint64`   | Сколько пар key-value в metadata                                        |

```go
// Header: offset 0..23, секция 1
// Логика: счётчики в header позволяют заранее выделить память под metadata и tensor info
type Header struct {
    Magic           [4]byte // offset 0, uint8[4], "GGUF" - первая проверка парсера
    Version         uint32 // offset 4; 2|3, v1 reject - определяет ширину полей и endianness
    TensorCount     uint64 // offset 8 - сколько раз читать tensor info entry
    MetadataKVCount uint64 // offset 16 - сколько раз читать KV; идёт сразу после header
}
```

Порядок байт - по разделу 3.2.

### 3.2. Определение endianness (v3)

По умолчанию GGUF - **little-endian**.

В v3 добавлена поддержка big-endian. Отдельного поля "byte order" нет: порядок определяется по последнему байту поля `version`.

| Шаг | Действие                | Описание                                      |
|-----|-------------------------|-----------------------------------------------|
| 1   | `read magic[4]`         | Проверка сигнатуры `"GGUF"`                   |
| 2   | `seek +3`               | Переход к байту [7] (4-й байт поля `version`) |
| 3   | `marker = read uint8`   | Эвристика endianness                          |
| 4   | `marker != 0`           | Файл big-endian                               |
| 5   | `seek -4`               | Возврат к началу поля `version` (offset 4)    |
| 6   | `version = read uint32` | Чтение версии с выбранным byte order          |

| `version` | байты [4..7]  | marker [7] | byte order | Описание               |
|-----------|---------------|------------|------------|------------------------|
| 2         | `02 00 00 00` | `0x00`     | LE         | Стандартный LE-файл v2 |
| 3         | `03 00 00 00` | `0x00`     | LE         | Стандартный LE-файл v3 |
| 3         | `00 00 00 03` | `0x03`     | BE         | BE-файл v3             |

v2: всегда little-endian.

```go
// detectEndianness: эвристика v3 по байту [7] поля version (раздел 3.2)
// Логика: в LE version=3 даёт байты 03 00 00 00, в BE - 00 00 00 03; по [7] отличаем
// v2 всегда LE - marker будет 0x00, ветка BE не сработает
func detectEndianness(r io.ReadSeeker) (binary.ByteOrder, error) {
    // magic[4] уже прочитан; курсор на offset 4 - начало version
    if _, err := r.Seek(3, io.SeekCurrent); err != nil { // seek -> offset 7
        return nil, err
    }

    var marker int8
    // marker читаем как LE uint8: нам нужен сырой байт [7], не значение version целиком
    if err := binary.Read(r, binary.LittleEndian, &marker); err != nil {
        return nil, err
    }

    order := binary.LittleEndian // default по спецификации
    if marker != 0 { // marker=0x03 => BE v3
        order = binary.BigEndian // ненулевой marker - признак BE-раскладки version
    }

    if _, err := r.Seek(-4, io.SeekCurrent); err != nil { // seek -> offset 4
        return nil, err
    }

    // откат: дальше version, tensor_count и все KV читаются с выбранным byte order

    return order, nil
}
```

Endianness применяется ко всем полям секций 1–3.

Внутри квант-блоков (раздел 10) scale и quants могут храниться в фиксированном порядке (обычно LE) - это свойство `ggml_type`, а не контейнера.

---

## 4. Примитивы

### 4.1. Целые и float

Все многобайтовые типы читаются в byte order файла (раздел 3.2).

| Тип                            | Size | Описание                              |
|--------------------------------|------|---------------------------------------|
| `uint8` / `int8`               | 1    | Беззнаковое / знаковое 8-битное целое |
| `uint16` / `int16`             | 2    | 16-битное целое                       |
| `uint32` / `int32` / `float32` | 4    | 32-битное целое или IEEE 754 float    |
| `uint64` / `int64` / `float64` | 8    | 64-битное целое или IEEE 754 double   |

```go
// readUint32/readUint64 - обёртки binary.Read с byte order файла (раздел 3.2)
// Логика: порядок байт един для всех полей секций 1-3; тип фиксирует размер чтения
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

// readScalar читает payload скалярного value_type (не ARRAY, не вложенный STRING в ARRAY)
// Логика: switch по ValueType - каждый case читает ровно sizeof(T) байт из потока
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
        var v uint8 // в файле bool - 0 или 1, не Go bool
        err := binary.Read(r, order, &v)
        return v == 1, err
    case String:
        return readString(r, order) // отдельная gguf_string
    case Uint64:
        return readUint64(r, order)
    // ... остальные типы 2,3,10,11,12 аналогично
    default:
        return nil, fmt.Errorf("unknown value type: %d", vtype)
    }
}
```

### 4.2. Строка (`gguf_string`)

Строки **не null-terminated** - длина задаётся явно.

| Поле   | Type     | Описание                              |
|--------|----------|---------------------------------------|
| `len`  | `uint64` | Длина строки в байтах (не символах)   |
| `data` | `[]byte` | Тело строки в UTF-8, без `\0` в конце |

```go
// gguf_string: uint64 len + len байт UTF-8, не null-terminated
// Логика: сначала узнаём размер, затем читаем ровно len байт - без поиска \0
type String struct {
    Len  uint64 // payload length в байтах (не в рунах Unicode)
    Data []byte // UTF-8 data[len]
}

// readString декодирует одну gguf_string из потока
func readString(r io.Reader, order binary.ByteOrder) (string, error) {
    length, err := readUint64(r, order) // поле len - всегда uint64 в v2/v3
    if err != nil {
        return "", err
    }

    data := make([]byte, length) // точный размер: лишние байты не читаем
    if _, err := io.ReadFull(r, data); err != nil {
        return "", err // EOF раньше len - файл обрезан или len повреждён
    }

    return string(data), nil // Go string поверх UTF-8 без копирования семантики \0
}
```

Ограничения ключей метаданных: ASCII, сегменты `lower_snake_case` через `.`, `len` <= 65535.  
Имена тензоров: `len` <= 64. Примеры: `token_embd.weight`, `blk.0.attn_q.weight`.

---

## 5. Metadata KV

Метаданные - самоописываемый словарь: каждое значение предваряется `value_type`.

### 5.1. Запись

| Поле         | Type          | Описание                                    |
|--------------|---------------|---------------------------------------------|
| `key`        | `gguf_string` | Уникальный иерархический идентификатор поля |
| `value_type` | `uint32`      | Тип значения (enum раздел 5.2)              |
| `value`      | по типу       | Скаляр или массив                           |

Повторяется `metadata_kv_count` раз, сразу после header.

```go
// metadata KV entry: key + value_type + value
// Логика: value_type говорит парсеру, сколько и каких байт читать после ключа
// порядок фиксирован: сначала key, потом type, потом payload - без длины всей записи
type MetadataKV struct {
    Key       string // gguf_string key - иерархия через '.', напр. qwen3.block_count
    ValueType ValueType // uint32 gguf_metadata_value_type - дискриминатор payload
    Value     any // scalar | []any (ARRAY) - уже распарсенное значение
}
```

### 5.2. `ValueType`

| ID  | Name      | Payload       | Описание                            |
|-----|-----------|---------------|-------------------------------------|
| 0   | `UINT8`   | `uint8`       | 8-битное беззнаковое целое          |
| 1   | `INT8`    | `int8`        | 8-битное знаковое целое             |
| 2   | `UINT16`  | `uint16`      | 16-битное беззнаковое               |
| 3   | `INT16`   | `int16`       | 16-битное знаковое                  |
| 4   | `UINT32`  | `uint32`      | 32-битное беззнаковое               |
| 5   | `INT32`   | `int32`       | 32-битное знаковое                  |
| 6   | `FLOAT32` | `float32`     | 32-битный float                     |
| 7   | `BOOL`    | `uint8`       | Логическое: `0` = false, `1` = true |
| 8   | `STRING`  | `gguf_string` | Текстовая строка                    |
| 9   | `ARRAY`   | раздел 5.3    | Однородный массив элементов         |
| 10  | `UINT64`  | `uint64`      | 64-битное беззнаковое               |
| 11  | `INT64`   | `int64`       | 64-битное знаковое                  |
| 12  | `FLOAT64` | `float64`     | 64-битный float                     |

```go
// gguf_metadata_value_type, ID 0..12
// Логика: самоописываемый формат - тип значения хранится в файле, не в схеме парсера
// новые ключи metadata можно добавлять без смены version структуры
type ValueType uint32

const (
    Uint8 ValueType = iota // 0, payload: uint8
    Int8 // 1, payload: int8
    Uint16 // 2, payload: uint16
    Int16 // 3, payload: int16
    Uint32 // 4, payload: uint32 - часто block_count, head_count
    Int32 // 5, payload: int32 - tokenizer.ggml.token_type[]
    Float32 // 6, payload: float32 - epsilon, rope.freq_base
    Bool // 7, payload: uint8 {0,1} - не native Go bool в файле
    String // 8, payload: gguf_string - general.architecture и т.п.
    Array // 9, payload: element_type+length+data - tokens, merges
    Uint64 // 10, payload: uint64 - большие размерности
    Int64 // 11, payload: int64
    Float64 // 12, payload: float64
)
```

### 5.3. Массив (`ARRAY`)

| Поле           | Type     | Описание                                   |
|----------------|----------|--------------------------------------------|
| `element_type` | `uint32` | Тип каждого элемента (0..12, не ARRAY)     |
| `length`       | `uint64` | Число элементов (не байт)                  |
| `payload`      | `[]T`    | Последовательность значений `element_type` |

```go
// ARRAY payload: element_type(uint32) + length(uint64) + elements[]
// Логика: массив вложен в value - сначала тип элементов, потом их количество, потом данные
type Array struct {
    ElementType ValueType // 0..12, != ARRAY - тип одного элемента payload
    Length      uint64 // element count - не байты, а число значений
    Elements    []any // распакованные элементы; строки - отдельные gguf_string
}

// readMetaValue читает одно поле value после value_type в KV-записи
func readMetaValue(r io.Reader, order binary.ByteOrder) (any, error) {
    vtype, err := readUint32(r, order) // value_type - первое поле value
    if err != nil {
        return nil, err
    }

    if ValueType(vtype) != Array {
        return readScalar(ValueType(vtype), r, order) // скаляр: фиксированный размер payload
    }

    // ветка ARRAY: структура value шире - нужны element_type и length
    elemType, err := readUint32(r, order) // ARRAY.element_type
    if err != nil {
        return nil, err
    }

    length, err := readUint64(r, order) // ARRAY.length - сколько элементов следом
    if err != nil {
        return nil, err
    }

    out := make([]any, length) // prealloc по length из файла
    for i := uint64(0); i < length; i++ {
        if ValueType(elemType) == String {
            // tokenizer.ggml.tokens: каждый элемент - своя len+data, не один blob
            out[i], err = readString(r, order)
        } else {
            out[i], err = readScalar(ValueType(elemType), r, order)
        }
        if err != nil {
            return nil, err // прерываем на первой битой записи
        }
    }

    return out, nil // []any - единый контейнер для любого типа metadata
}
```

Для `STRING`: каждый элемент - отдельная `gguf_string`.

Вложенные массивы: `element_type = ARRAY`.

### 5.4. Схема ключей

| Префикс                  | Описание                                           |
|--------------------------|----------------------------------------------------|
| `general.<field>`        | Свойства файла и модели в целом                    |
| `<architecture>.<field>` | Гиперпараметры архитектуры (`qwen3`, `llama`, ...) |
| `tokenizer.<field>`      | Словарь, BPE merges, chat template                 |
| `<community>.<field>`    | Произвольные расширения с namespace                |

`<architecture>` = значение `general.architecture`. Один файл - одна архитектура.

---

## 6. Стандартные ключи метаданных

### 6.1. Обязательные

| Key                            | Type   | Описание                                                       |
|--------------------------------|--------|----------------------------------------------------------------|
| `general.architecture`         | string | Имя архитектуры; определяет префикс `<arch>.*` и загрузчик     |
| `general.quantization_version` | uint32 | Версия схемы квантизации; обязателен при квантованных тензорах |
| `general.alignment`            | uint32 | Выравнивание данных; default **32**; кратно 8                  |

### 6.2. `general.*`

| Key                    | Type     | Описание                                |
|------------------------|----------|-----------------------------------------|
| `general.name`         | string   | Человекочитаемое имя модели             |
| `general.author`       | string   | Автор или команда                       |
| `general.version`      | string   | Версия модели (напр. `v1.0`)            |
| `general.organization` | string   | Организация-разработчик                 |
| `general.basename`     | string   | Базовое имя архитектуры без fine-tune   |
| `general.finetune`     | string   | Тип дообучения: Chat, Instruct, ...     |
| `general.description`  | string   | Текстовое описание модели               |
| `general.size_label`   | string   | Класс размера: `8B`, `0.6B`, `8x7B`     |
| `general.file_type`    | uint32   | Доминирующий тип квантизации (раздел 7) |
| `general.license`      | string   | Лицензия в формате SPDX                 |
| `general.url`          | string   | URL страницы модели                     |
| `general.tags`         | string[] | Теги для поиска и каталогизации         |
| `general.languages`    | string[] | Поддерживаемые языки (ISO 639)          |

### 6.3. `<arch>.*`

Гиперпараметры transformer.

Числа могут быть `uint32` или `uint64`.

| Key                                       | Type            | Описание                                   |
|-------------------------------------------|-----------------|--------------------------------------------|
| `<arch>.context_length`                   | uint32 \ uint64 | Максимальное число токенов в контексте     |
| `<arch>.embedding_length`                 | uint32 \ uint64 | Размерность embedding (hidden size)        |
| `<arch>.block_count`                      | uint32 \ uint64 | Число transformer-блоков (слоёв)           |
| `<arch>.feed_forward_length`              | uint32 \ uint64 | Размерность intermediate слоя FFN          |
| `<arch>.attention.head_count`             | uint32 \ uint64 | Число query-heads в attention              |
| `<arch>.attention.head_count_kv`          | uint32 \ uint64 | Число key/value heads (GQA/MQA)            |
| `<arch>.attention.key_length`             | uint32 \ uint64 | Размерность K/V на один head (head_dim)    |
| `<arch>.attention.layer_norm_rms_epsilon` | float32         | Epsilon в RMSNorm                          |
| `<arch>.rope.freq_base`                   | float32         | Базовая частота Rotary Position Embedding  |
| `<arch>.rope.freq_scale`                  | float32         | Масштабный коэффициент RoPE (long context) |

### 6.4. `tokenizer.*`

| Key                               | Type     | Описание                                          |
|-----------------------------------|----------|---------------------------------------------------|
| `tokenizer.ggml.model`            | string   | Семейство tokenizer: `gpt2`, `llama`, ...         |
| `tokenizer.ggml.pre`              | string   | Pretokenizer: `default`, `qwen2`, `llama3`, ...   |
| `tokenizer.ggml.tokens`           | string[] | Полный словарь: индекс -> строка токена           |
| `tokenizer.ggml.merges`           | string[] | BPE merge rules: `"token_a token_b"`              |
| `tokenizer.ggml.token_type`       | int32[]  | Тип каждого токена: normal, control, user-defined |
| `tokenizer.ggml.bos_token_id`     | uint32   | ID токена начала последовательности               |
| `tokenizer.ggml.eos_token_id`     | uint32   | ID токена конца / stop                            |
| `tokenizer.ggml.padding_token_id` | uint32   | ID padding-токена                                 |
| `tokenizer.ggml.add_bos_token`    | bool     | Автодобавление BOS при encode                     |
| `tokenizer.ggml.add_eos_token`    | bool     | Автодобавление EOS при encode                     |
| `tokenizer.chat_template`         | string   | Jinja2-шаблон для форматирования диалога          |

---

## 7. `general.file_type`

Сводный индикатор доминирующей квантизации файла.

| ID  | Name                   | Описание                                     |
|-----|------------------------|----------------------------------------------|
| 0   | `ALL_F32`              | Все тензоры в FP32                           |
| 1   | `MOSTLY_F16`           | Большинство тензоров FP16                    |
| 2   | `MOSTLY_Q4_0`          | Большинство тензоров Q4_0 (4-bit, symmetric) |
| 3   | `MOSTLY_Q4_1`          | Большинство тензоров Q4_1 (4-bit, with min)  |
| 4   | `MOSTLY_Q4_1_SOME_F16` | Q4_1 + отдельные FP16 тензоры                |
| 5   | `MOSTLY_Q4_2`          | Устарел; удалён из llama.cpp                 |
| 6   | `MOSTLY_Q4_3`          | Устарел; удалён из llama.cpp                 |
| 7   | `MOSTLY_Q8_0`          | Большинство тензоров Q8_0 (8-bit)            |
| 8   | `MOSTLY_Q5_0`          | Большинство тензоров Q5_0 (5-bit)            |
| 9   | `MOSTLY_Q5_1`          | Большинство тензоров Q5_1                    |
| 10  | `MOSTLY_Q2_K`          | Большинство тензоров Q2_K (K-quant, 2-bit)   |
| 11  | `MOSTLY_Q3_K_S`        | Q3_K small                                   |
| 12  | `MOSTLY_Q3_K_M`        | Q3_K medium                                  |
| 13  | `MOSTLY_Q3_K_L`        | Q3_K large                                   |
| 14  | `MOSTLY_Q4_K_S`        | Q4_K small                                   |
| 15  | `MOSTLY_Q4_K_M`        | Q4_K medium                                  |
| 16  | `MOSTLY_Q5_K_S`        | Q5_K small                                   |
| 17  | `MOSTLY_Q5_K_M`        | Q5_K medium                                  |
| 18  | `MOSTLY_Q6_K`          | Большинство тензоров Q6_K                    |

---

## 8. Tensor Info

Tensor Info - каталог весов.

Каждая запись описывает один тензор; размер в байтах вычисляется, не хранится.

Повторяется `tensor_count` раз, после metadata.

| Поле           | Type          | Описание                                             |
|----------------|---------------|------------------------------------------------------|
| `name`         | `gguf_string` | Уникальное имя тензора (`blk.0.attn_q.weight`, ...)  |
| `n_dimensions` | `uint32`      | Ранг тензора (число измерений)                       |
| `dimensions`   | `uint64[n]`   | Размер по каждой оси; произведение = число элементов |
| `type`         | `uint32`      | Тип хранения (`ggml_type`, раздел 9)                 |
| `offset`       | `uint64`      | Смещение от `tensor_data_start`; кратно `ALIGNMENT`  |

| Field           | Constraint         | Описание                              |
|-----------------|--------------------|---------------------------------------|
| `n_dimensions`  | >= 1               | Ранг тензора; в спецификации <= 4     |
| `dimensions[i]` | > 0                | Каждая ось имеет положительный размер |
| `offset`        | `% ALIGNMENT == 0` | Начало тензора на выровненной границе |

```go
// tensor info entry (секция 3): name + n_dimensions + dimensions[] + type + offset
// Логика: дескриптор не содержит весов - только как их найти и интерпретировать в tensor data порядок полей в файле строгий; размер тензора в байтах здесь не хранится
type TensorInfo struct {
    Name       string // gguf_string name - уникальный идентификатор, напр. blk.0.attn_q.weight
    Dimensions []uint64 // uint32 n_dimensions + uint64[n] - произведение = число элементов
    Type       GGMLType // uint32 ggml_type - выбирает декодер и формулу tensorSize
    Offset     uint64 // от tensor_data_start, % ALIGNMENT == 0
}

// size_bytes = ceil(prod(dimensions) / block_values) * block_bytes
// Логика: размер в файле не записан - считаем по числу элементов и параметрам квант-блока
func tensorSize(typ GGMLType, dims []uint64) int64 {
    n := uint64(1)
    for _, d := range dims {
        n *= d // prod(dimensions) - общее число весов тензора
    }

    bv, bb := blockParams(typ) // block_values, block_bytes - зависят только от ggml_type
    blocks := (n + bv - 1) / bv // ceil(n / bv) - хвостовой неполный блок тоже целый

    return int64(blocks * bb) // итог в байтах для SectionReader и валидации границ
}
```

Конвенция GGML для 2D-матриц: `dimensions = [cols, rows]` (вход * выход).

---

## 9. `ggml_type`

Тип определяет интерпретацию сырых байт.

Квантизованные типы хранят веса блоками с общим scale.

| ID  | Name      | `block_values` | `block_bytes` | Описание                                 |
|-----|-----------|----------------|---------------|------------------------------------------|
| 0   | `F32`     | 1              | 4             | Полная точность IEEE 754                 |
| 1   | `F16`     | 1              | 2             | Половинная точность                      |
| 2   | `Q4_0`    | 32             | 18            | 4-bit symmetric, 32 веса/блок            |
| 3   | `Q4_1`    | 32             | 20            | 4-bit с min, 32 веса/блок                |
| -   | `Q4_2`    | -              | -             | Удалён (id 4)                            |
| -   | `Q4_3`    | -              | -             | Удалён (id 5)                            |
| 6   | `Q5_0`    | 32             | 22            | 5-bit symmetric                          |
| 7   | `Q5_1`    | 32             | 26            | 5-bit с min                              |
| 8   | `Q8_0`    | 32             | 34            | 8-bit symmetric; ~4* сжатие от F32       |
| 9   | `Q8_1`    | 32             | 36            | 8-bit + поле sum для dot-product         |
| 10  | `Q2_K`    | 256            | 84            | K-quant 2-bit, блок 256                  |
| 11  | `Q3_K`    | 256            | 110           | K-quant 3-bit                            |
| 12  | `Q4_K`    | 256            | 144           | K-quant 4-bit                            |
| 13  | `Q5_K`    | 256            | 176           | K-quant 5-bit                            |
| 14  | `Q6_K`    | 256            | 210           | K-quant 6-bit                            |
| 15  | `Q8_K`    | 256            | 292           | K-quant 8-bit                            |
| 16  | `IQ2_XXS` | -              | var           | Importance quant 2-bit extra-extra-small |
| 17  | `IQ2_XS`  | -              | var           | Importance quant 2-bit extra-small       |
| 18  | `IQ3_XXS` | -              | var           | Importance quant 3-bit extra-extra-small |
| 19  | `IQ1_S`   | -              | var           | Importance quant 1-bit small             |
| 20  | `IQ4_NL`  | -              | var           | Importance quant 4-bit non-linear        |
| 21  | `IQ3_S`   | -              | var           | Importance quant 3-bit small             |
| 22  | `IQ2_S`   | -              | var           | Importance quant 2-bit small             |
| 23  | `IQ4_XS`  | -              | var           | Importance quant 4-bit extra-small       |
| 24  | `I8`      | 1              | 1             | Целое 8-bit                              |
| 25  | `I16`     | 1              | 2             | Целое 16-bit                             |
| 26  | `I32`     | 1              | 4             | Целое 32-bit                             |
| 27  | `I64`     | 1              | 8             | Целое 64-bit                             |
| 28  | `F64`     | 1              | 8             | Double precision                         |
| 29  | `IQ1_M`   | -              | var           | Importance quant 1-bit medium            |
| 30  | `BF16`    | 1              | 2             | Brain float 16                           |
| 34  | `TQ1_0`   | -              | var           | Ternary quant 1.0                        |
| 35  | `TQ2_0`   | -              | var           | Ternary quant 2.0                        |
| 39  | `MXFP4`   | -              | var           | Microscaling FP4                         |

```go
// ggml_type (uint32), определяет layout tensor data
// Логика: один и тот же срез байт читается по-разному в зависимости от ggml_type ID в enum совпадает с числовым значением в поле type тензора
type GGMLType uint32

const (
    F32 GGMLType = iota // 0, block_values=1, block_bytes=4 - без квантизации
    F16 // 1, block_values=1, block_bytes=2
    Q4_0 // 2, block_values=32, block_bytes=18 - 4-bit symmetric
    Q4_1 // 3, block_values=32, block_bytes=20 - 4-bit + min
    _ // 4, Q4_2 deprecated - id зарезервирован, пропускаем в enum
    _ // 5, Q4_3 deprecated - иначе сместятся ID последующих типов
    Q5_0 // 6, block_values=32, block_bytes=22
    Q5_1 // 7, block_values=32, block_bytes=26
    Q8_0 // 8, block_values=32, block_bytes=34 - основной тип в gguf.go
    Q8_1 // 9, block_values=32, block_bytes=36 - + sum для matmul
    Q2_K // 10, block_values=256, block_bytes=84 - K-quant family
    Q3_K // 11, block_values=256, block_bytes=110
    Q4_K // 12, block_values=256, block_bytes=144
    Q5_K // 13, block_values=256, block_bytes=176
    Q6_K // 14, block_values=256, block_bytes=210
    Q8_K // 15, block_values=256, block_bytes=292
    // ...
)

// blockParams возвращает (block_values, block_bytes) для ggml_type
// Логика: lookup-таблица из раздела 9; F32/I8 - 1 значение на блок, Q*_0 - 32, Q*_K - 256
func blockParams(typ GGMLType) (uint64, uint64) {
    switch typ {
    case F32:
        return 1, 4 // block_values=1, block_bytes=4
    case Q8_0:
        return 32, 34
    case Q4_0:
        return 32, 18
    // ... остальные типы из таблицы раздела 9
    default:
        return 0, 0
    }
}
```

---

## 10. Layout квант-блоков

Квант-блок - минимальная единица деквантизации.

### 10.1. Q4_0 - 18 байт, 32 значения

| Offset | Size | Field | Type     | Описание                       |
|--------|------|-------|----------|--------------------------------|
| 0      | 2    | `d`   | float16  | Масштаб блока                  |
| 2      | 16   | `qs`  | `[]byte` | 32 * 4-bit, по 2 nibble в байт |

```go
// Q4_0 block: d(float16) + qs[16]; 18 байт -> 32 float32
// w[i] = d * (q[i] - 8), zero-point=8
// Логика: один scale d на 32 веса; 4-битные q упакованы по 2 в байт; симметрия вокруг 8
func dequantQ4_0(block []byte) ([32]float32, error) {
    d := fp16ToF32(binary.LittleEndian.Uint16(block[0:2])) // offset 0, field d - общий масштаб
	
    var out [32]float32 // ровно block_values для Q4_0
    for i := 0; i < 32; i++ {
        // i%2==0: младшие 4 бита байта; i%2==1: старшие 4 бита того же байта
        q := (block[2+i/2] >> (4 * (i % 2))) & 0x0F // qs: 2 nibble/byte
        out[i] = d * float32(int(q)-8) // zero-point 8 центрирует диапазон 0..15
    }
	
    return out, nil // 32 восстановленных float32 для matmul/dequant pipeline
}
```

### 10.2. Q4_1 - 20 байт, 32 значения

| Offset | Size | Field | Type     | Описание       |
|--------|------|-------|----------|----------------|
| 0      | 2    | `d`   | float16  | Масштаб        |
| 2      | 2    | `m`   | float16  | Смещение (min) |
| 4      | 16   | `qs`  | `[]byte` | 32 * 4-bit     |

```go
// Q4_1 block: d(float16) + m(float16) + qs[16]; 20 байт -> 32 float32
// w[i] = d * q[i] + m
// Логика: в отличие от Q4_0, диапазон задаётся линейно через min (m), без zero-point
func dequantQ4_1(block []byte) ([32]float32, error) {
    d := fp16ToF32(binary.LittleEndian.Uint16(block[0:2])) // offset 0, field d - шаг шкалы
    m := fp16ToF32(binary.LittleEndian.Uint16(block[2:4])) // offset 2, field m - смещение min

    var out [32]float32
    for i := 0; i < 32; i++ {
        q := (block[4+i/2] >> (4 * (i % 2))) & 0x0F // offset 4, field qs - упаковка как в Q4_0
        out[i] = d*float32(q) + m // affine: q∈[0,15] растягивается, m задаёт базу
    }
	
    return out, nil
}
```

### 10.3. Q5_0 - 22 байта, 32 значения

| Offset | Size | Field | Type      | Описание                      |
|--------|------|-------|-----------|-------------------------------|
| 0      | 2    | `d`   | float16   | Масштаб                       |
| 2      | 4    | `qh`  | uint8[4]  | Старшие биты 5-bit квантов    |
| 6      | 16   | `qs`  | uint8[16] | Младшие 4 бита каждого кванта |

### 10.4. Q8_0 - 34 байта, 32 значения

| Offset | Size | Field | Type     | Описание                 |
|--------|------|-------|----------|--------------------------|
| 0      | 2    | `d`   | float16  | Масштаб блока            |
| 2      | 32   | `qs`  | `[]int8` | 32 знаковых 8-bit кванта |

```go
// Q8_0 block: d(float16) + qs[int8;32]; 34 байт -> 32 float32
// w[i] = d * qs[i]; d обычно LE внутри блока
// Логика: каждый вес - знаковый int8; точность выше Q4, но блок почти в 2 раза больше scale d внутри блока - свойство ggml_type, не byte order контейнера GGUF
func dequantQ8_0(block []byte) ([32]float32, error) {
    d := fp16ToF32(binary.LittleEndian.Uint16(block[0:2])) // offset 0, field d
    var out [32]float32
    for i := 0; i < 32; i++ {
        out[i] = d * float32(int8(block[2+i])) // qs[i] - отдельный байт, без упаковки nibble
    }

    return out, nil
}
```

### 10.5. Q8_1 - 36 байт, 32 значения

| Offset | Size | Field | Type     | Описание                                   |
|--------|------|-------|----------|--------------------------------------------|
| 0      | 2    | `d`   | float16  | Масштаб                                    |
| 2      | 2    | `s`   | float16  | Предвычисленная сумма квантов (для matmul) |
| 4      | 32   | `qs`  | int8[32] | 32 знаковых кванта                         |

### 10.6. K-кванты (Q2_K ... Q8_K)

`block_values = 256`. Внутри: sub-block scales, битовые маски, упакованные nibble. `block_bytes` фиксирован (раздел 9).

### 10.7. F32 / F16 / I8 / I16 / I32

Прямое хранение без деквантизации: один элемент = один блок.

---

## 11. Tensor Data

| Элемент          | Описание                                           |
|------------------|----------------------------------------------------|
| Padding          | Нули `0x00` от конца tensor info до `align_up`     |
| `tensor_data`    | Непрерывная зона сырых байт всех тензоров          |
| `T.offset`       | Начало тензора внутри `tensor_data`                |
| Межтензорный gap | Padding `0x00` до следующего выровненного `offset` |

Данные могут отличаться от исходной модели из-за квантизации или транспонирования - интерпретация задаётся tensor info + metadata.

```go
// padding 0x00: seek от pos до align_up(pos, align)
// Логика: writer дописывает нули до границы; reader перепрыгивает их, не парся как данные pos - конец tensor info; dataStart - абсолютное смещение начала tensor data в файле
func skipPadding(r io.ReadSeeker, pos, align int64) (int64, error) {
    dataStart := alignUp(pos, align) // tensor_data_start
    if _, err := r.Seek(dataStart, io.SeekStart); err != nil {
        return 0, err // seek за пределы файла - pos/align некорректны
    }

    return dataStart, nil // возвращаем вычисленный offset для TensorOffset
}
```

---

## 12. Алгоритм парсинга

Два этапа: (1) описательная часть, (2) ленивый доступ к весам.

| Шаг | Действие               | Описание                            |
|-----|------------------------|-------------------------------------|
| 1   | Проверить `magic`      | Валидация формата                   |
| 2   | Определить byte order  | раздел 3.2                          |
| 3   | Прочитать `version`    | Принять 2 или 3                     |
| 4   | Прочитать счётчики     | `tensor_count`, `metadata_kv_count` |
| 5   | Разобрать metadata     | `metadata_kv_count` пар KV          |
| 6   | Получить `alignment`   | Из metadata, default 32             |
| 7   | Разобрать tensor info  | `tensor_count` дескрипторов         |
| 8   | Вычислить `data_start` | `align_up` текущей позиции          |
| 9   | Читать веса по запросу | `slice(data_start + offset, size)`  |

```go
// Parse: секции 1-3 (header, metadata, tensor info); tensor data - lazy
// Логика: двухфазный парсинг - сначала вся "карта" файла, веса читаются по требованию после Parse курсор на tensor_data_start, но байты весов в File не копируются
func Parse(r io.ReadSeeker) (*File, error) {
    // --- шаг 1: magic ---
    var magic [4]byte
    if _, err := io.ReadFull(r, magic[:]); err != nil {
        return nil, err
    }

    if !bytes.Equal(magic[:], []byte("GGUF")) {
        return nil, fmt.Errorf("not a GGUF file") // ранний выход: не тратим время на разбор
    }

    // шаг 2: byte order (до чтения version и всех uint64)
    byteOrder, err := detectEndianness(r) // v3: LE|BE; v2: всегда LE
    if err != nil {
        return nil, err
    }

    // шаг 3: version
    version, err := readUint32(r, byteOrder)
    if err != nil {
        return nil, err
    }
    if version != 2 && version != 3 {
        return nil, fmt.Errorf("unsupported version: %d (v1 not supported)", version)
    }

    // шаг 4: счётчики из header (уже после magic+version в потоке)
    tensorCount, err := readUint64(r, byteOrder) // сколько tensor info entries
    if err != nil {
        return nil, err
    }

    metadataCount, err := readUint64(r, byteOrder) // сколько KV-пар в metadata
    if err != nil {
        return nil, err
    }

    // шаг 5: metadata KV (секция 2)
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

        metadata[key] = val // map даёт O(1) доступ к general.* и <arch>.*
    }

    // шаг 6: ALIGNMENT из metadata
    alignment := int64(32)
    if v, ok := metadata["general.alignment"].(uint32); ok {
        alignment = int64(v) // иначе padding и tensor offsets не совпадут с writer
    }

    // шаг 7: tensor info (секция 3)
    tensors := make([]TensorInfo, tensorCount)
    for i := uint64(0); i < tensorCount; i++ {
        name, err := readString(r, byteOrder) // name
        if err != nil {
            return nil, err
        }

        ndim, err := readUint32(r, byteOrder) // n_dimensions - ранг тензора
        if err != nil {
            return nil, err
        }

        dims := make([]uint64, ndim)
        for j := uint32(0); j < ndim; j++ {
            dims[j], err = readUint64(r, byteOrder) // dimensions[j] - размер оси j
            if err != nil {
                return nil, err
            }
        }

        typ, err := readUint32(r, byteOrder) // type (ggml_type)
        if err != nil {
            return nil, err
        }

        offset, err := readUint64(r, byteOrder) // offset - относительно tensor data
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

    // шаг 8: tensor_data_start
    pos, err := r.Seek(0, io.SeekCurrent) // конец tensor info - начало padding
    if err != nil {
        return nil, err
    }
    dataStart := alignUp(pos, alignment) // tensor_data_start; дальше - только веса

    return &File{
        Version:      int(version),
        ByteOrder:    byteOrder,
        Metadata:     metadata,
        Tensors:      tensors,
        TensorOffset: dataStart, // якорь для всех TensorData()
    }, nil
}

// slice [tensor_data_start+offset, tensor_data_start+offset+size)
// Логика: SectionReader не копирует гигабайты в RAM - читаем окно нужного тензора ReaderAt позволяет mmap-файл читать по смещению без последовательного seek
func (f *File) TensorData(r io.ReaderAt, t TensorInfo) (io.Reader, error) {
    start := f.TensorOffset + int64(t.Offset) // абсолютный offset в файле
    size := tensorSize(t.Type, t.Dimensions) // граница среза - только этот тензор
    return io.NewSectionReader(r, start, size), nil
}
```

---

## 13. Полная структура (Go)

| Структура / поле         | Описание                                 |
|--------------------------|------------------------------------------|
| `File`                   | Корневой layout всего файла              |
| `Header.Magic`           | Сигнатура `"GGUF"`                       |
| `Header.Version`         | Версия формата: 2 или 3                  |
| `Header.TensorCount`     | Число тензоров                           |
| `Header.MetadataKVCount` | Число KV-пар                             |
| `Metadata`               | `map[string]any` метаданных              |
| `Tensors`                | Срез дескрипторов тензоров               |
| `TensorOffset`           | Абсолютное смещение начала `tensor_data` |
| `MetadataKV.Key`         | Имя поля метаданных                      |
| `MetadataKV.ValueType`   | Тип значения                             |
| `MetadataKV.Value`       | Скаляр или массив                        |
| `TensorInfo.Name`        | Имя тензора                              |
| `TensorInfo.Dimensions`  | Размеры по осям                          |
| `TensorInfo.Type`        | `GGMLType`                               |
| `TensorInfo.Offset`      | Смещение в `tensor_data`                 |
| `String.Len`             | Длина строки в байтах                    |
| `String.Data`            | Тело строки UTF-8                        |

```go
// Логика: File - итог Parse(); связывает metadata, каталог тензоров и точку входа в веса без TensorOffset нельзя вычислить абсолютный адрес ни одного тензора
type File struct {
    Version      int // header.version - влияет на ширину полей и endianness
    ByteOrder    binary.ByteOrder // LE|BE, секции 1-3; квант-блоки могут быть LE внутри
    Metadata     map[string]any // metadata KV map - hyperparams, tokenizer, general.*
    Tensors      []TensorInfo // tensor info array - каталог без загрузки весов
    TensorOffset int64 // align_up(end_of_tensor_info) - база для всех offset
}

// Header - wire-формат первых 24 байт; в Parse читается по полям, не одним Read структуры
type Header struct {
    Magic           [4]byte // uint8[4] - сигнатура для валидации
    Version         uint32 // после magic; определяет reject v1 и detectEndianness
    TensorCount     uint64 // задаёт длину цикла чтения tensor info
    MetadataKVCount uint64 // задаёт длину цикла чтения metadata
}

// MetadataKV - логическая единица секции 2; в map хранится только Key -> Value
type MetadataKV struct {
    Key       string // gguf_string - уникальный идентификатор поля
    ValueType ValueType // uint32 - дискриминатор payload; без него нельзя прочитать value
    Value     any // распарсенное значение - скаляр или срез для ARRAY
}

// TensorInfo - логическая единица секции 3; веса лежат отдельно в tensor data
type TensorInfo struct {
    Name       string // gguf_string - ключ для поиска веса по имени слоя
    Dimensions []uint64 // uint64[n_dimensions] - форма; нужна для tensorSize
    Type       GGMLType // uint32 - выбор декодера байт (F32, Q8_0, ...)
    Offset     uint64 // % ALIGNMENT == 0 - смещение от TensorOffset
}

// String - вспомогательное представление gguf_string при чтении из потока
type String struct {
    Len  uint64 // uint64 - длина payload в байтах
    Data []byte // UTF-8[len], без \0 - ровно len байт из файла
}
```

---

## 14. Инварианты и ошибки

| Условие                           | Действие   | Описание                            |
|-----------------------------------|------------|-------------------------------------|
| `magic != "GGUF"`                 | reject     | Файл не является GGUF               |
| `version == 1`                    | reject     | v1 не поддерживается                |
| `version not in {2, 3}`           | reject     | Неподдерживаемая версия структуры   |
| `BOOL not in {0, 1}`              | reject     | Некорректное логическое значение    |
| `value_type` неизвестен           | reject     | Парсер не может прочитать значение  |
| `T.offset % ALIGNMENT != 0`       | reject     | Нарушено выравнивание тензора       |
| `general.alignment % 8 != 0`      | reject     | Невалидное значение alignment       |
| нет `general.architecture`        | incomplete | Неизвестна архитектура модели       |
| кванты без `quantization_version` | incomplete | Неизвестна версия схемы квантизации |

Читатель должен принимать `uint32` и `uint64` для числовых полей метаданных.

---

## 15. v1 (не поддерживается)

Формат v1 в современных парсерах **не читается**. Файл с `version == 1` отклоняется до разбора metadata и tensor info.

Отличия v1 от v2 (справочно, для идентификации устаревших файлов):

| Поле                | v1       | Описание                   |
|---------------------|----------|----------------------------|
| `tensor_count`      | `uint32` | 32-битный счётчик тензоров |
| `metadata_kv_count` | `uint32` | 32-битный счётчик KV       |
| `dimensions[i]`     | `uint32` | 32-битные размерности      |
| endianness          | LE only  | Только little-endian       |
