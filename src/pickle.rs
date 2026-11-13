//! Just enough of the pickle VM to read a torch state_dict.
//!
//! torch saves `OrderedDict[str, Tensor]` where every tensor is
//! `torch._utils._rebuild_tensor_v2(storage, offset, size, stride, ...)` and the
//! storage comes from a persistent id tuple
//! `('storage', torch.BFloat16Storage, '0', 'cpu', numel)`.

use std::collections::HashMap;

#[derive(Clone, Debug)]
pub enum Val {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Bytes(Vec<u8>),
    Tuple(Vec<Val>),
    List(Vec<Val>),
    Dict(Vec<(Val, Val)>),
    Set(Vec<Val>),
    Global(String),
    Storage {
        dtype: String,
        key: String,
        numel: i64,
    },
    Tensor(Tensor),
    Other(String),
}

#[derive(Clone, Debug)]
pub struct Tensor {
    pub dtype: String,
    pub key: String,
    pub storage_numel: i64,
    pub offset: i64,
    pub size: Vec<i64>,
    pub stride: Vec<i64>,
}

pub fn parse(data: &[u8]) -> Result<Val, String> {
    let mut stack: Vec<Val> = Vec::new();
    let mut marks: Vec<usize> = Vec::new();
    let mut memo: HashMap<usize, Val> = HashMap::new();
    let mut i = 0usize;

    let pop = |stack: &mut Vec<Val>| -> Result<Val, String> { stack.pop().ok_or_else(|| "pickle stack underflow".to_string()) };
    let pop_to_mark = |stack: &mut Vec<Val>, marks: &mut Vec<usize>| -> Result<Vec<Val>, String> {
        let m = marks.pop().ok_or_else(|| "pickle mark underflow".to_string())?;
        if m > stack.len() {
            return Err("pickle mark out of range".into());
        }
        Ok(stack.split_off(m))
    };

    loop {
        if i >= data.len() {
            return Err("unexpected end of pickle".into());
        }
        let op = data[i];
        i += 1;
        match op {
            0x80 => {
                // PROTO
                i += 1;
            }
            0x2e => {
                // STOP
                return stack.pop().ok_or_else(|| "empty stack at STOP".to_string());
            }
            0x28 => marks.push(stack.len()),
            0x29 => stack.push(Val::Tuple(Vec::new())),
            0x74 => {
                // TUPLE
                let items = pop_to_mark(&mut stack, &mut marks)?;
                stack.push(Val::Tuple(items));
            }
            0x85 => {
                let a = pop(&mut stack)?;
                stack.push(Val::Tuple(vec![a]));
            }
            0x86 => {
                let b = pop(&mut stack)?;
                let a = pop(&mut stack)?;
                stack.push(Val::Tuple(vec![a, b]));
            }
            0x87 => {
                let c = pop(&mut stack)?;
                let b = pop(&mut stack)?;
                let a = pop(&mut stack)?;
                stack.push(Val::Tuple(vec![a, b, c]));
            }
            0x7d => stack.push(Val::Dict(Vec::new())),
            0x5d => stack.push(Val::List(Vec::new())),
            0x8f => stack.push(Val::Set(Vec::new())),
            0x73 => {
                // SETITEM
                let v = pop(&mut stack)?;
                let k = pop(&mut stack)?;
                match stack.last_mut() {
                    Some(Val::Dict(d)) => d.push((k, v)),
                    _ => return Err("SETITEM on a non-dict".into()),
                }
            }
            0x75 => {
                // SETITEMS
                let items = pop_to_mark(&mut stack, &mut marks)?;
                match stack.last_mut() {
                    Some(Val::Dict(d)) => {
                        if items.len() % 2 != 0 {
                            return Err("SETITEMS with odd item count".into());
                        }
                        let mut it = items.into_iter();
                        while let (Some(k), Some(v)) = (it.next(), it.next()) {
                            d.push((k, v));
                        }
                    }
                    _ => return Err("SETITEMS on a non-dict".into()),
                }
            }
            0x61 => {
                // APPEND
                let v = pop(&mut stack)?;
                match stack.last_mut() {
                    Some(Val::List(l)) => l.push(v),
                    _ => return Err("APPEND on a non-list".into()),
                }
            }
            0x65 => {
                // APPENDS
                let items = pop_to_mark(&mut stack, &mut marks)?;
                match stack.last_mut() {
                    Some(Val::List(l)) => l.extend(items),
                    _ => return Err("APPENDS on a non-list".into()),
                }
            }
            0x90 => {
                // ADDITEMS
                let items = pop_to_mark(&mut stack, &mut marks)?;
                match stack.last_mut() {
                    Some(Val::Set(s)) => s.extend(items),
                    _ => return Err("ADDITEMS on a non-set".into()),
                }
            }
            0x58 => {
                // BINUNICODE
                if i + 4 > data.len() {
                    return Err("truncated BINUNICODE".into());
                }
                let n = u32::from_le_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]) as usize;
                i += 4;
                let s = std::str::from_utf8(&data[i..i + n]).map_err(|_| "bad utf8 in pickle".to_string())?;
                stack.push(Val::Str(s.to_string()));
                i += n;
            }
            0x8c => {
                // SHORT_BINUNICODE
                let n = *data.get(i).ok_or("truncated SHORT_BINUNICODE")? as usize;
                i += 1;
                let s = std::str::from_utf8(&data[i..i + n]).map_err(|_| "bad utf8 in pickle".to_string())?;
                stack.push(Val::Str(s.to_string()));
                i += n;
            }
            0x42 => {
                let n = u32::from_le_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]) as usize;
                i += 4;
                stack.push(Val::Bytes(data[i..i + n].to_vec()));
                i += n;
            }
            0x43 => {
                let n = *data.get(i).ok_or("truncated SHORT_BINBYTES")? as usize;
                i += 1;
                stack.push(Val::Bytes(data[i..i + n].to_vec()));
                i += n;
            }
            0x4a => {
                // BININT
                let v = i32::from_le_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]);
                i += 4;
                stack.push(Val::Int(v as i64));
            }
            0x4b => {
                stack.push(Val::Int(*data.get(i).ok_or("truncated BININT1")? as i64));
                i += 1;
            }
            0x4d => {
                let v = u16::from_le_bytes([data[i], data[i + 1]]);
                i += 2;
                stack.push(Val::Int(v as i64));
            }
            0x8a => {
                // LONG1
                let n = *data.get(i).ok_or("truncated LONG1")? as usize;
                i += 1;
                let mut v: i128 = 0;
                for k in 0..n {
                    v |= (data[i + k] as i128) << (8 * k);
                }
                if n > 0 && data[i + n - 1] & 0x80 != 0 {
                    v -= 1i128 << (8 * n);
                }
                i += n;
                stack.push(Val::Int(v as i64));
            }
            0x47 => {
                // BINFLOAT (big endian)
                let mut b = [0u8; 8];
                b.copy_from_slice(&data[i..i + 8]);
                i += 8;
                stack.push(Val::Float(f64::from_be_bytes(b)));
            }
            0x4e => stack.push(Val::None),
            0x88 => stack.push(Val::Bool(true)),
            0x89 => stack.push(Val::Bool(false)),
            0x71 => {
                // BINPUT
                let idx = *data.get(i).ok_or("truncated BINPUT")? as usize;
                i += 1;
                let top = stack.last().cloned().ok_or("BINPUT on empty stack")?;
                memo.insert(idx, top);
            }
            0x72 => {
                // LONG_BINPUT
                let idx = u32::from_le_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]) as usize;
                i += 4;
                let top = stack.last().cloned().ok_or("LONG_BINPUT on empty stack")?;
                memo.insert(idx, top);
            }
            0x94 => {
                // MEMOIZE
                let idx = memo.len();
                let top = stack.last().cloned().ok_or("MEMOIZE on empty stack")?;
                memo.insert(idx, top);
            }
            0x68 => {
                let idx = *data.get(i).ok_or("truncated BINGET")? as usize;
                i += 1;
                stack.push(memo.get(&idx).cloned().ok_or("BINGET with unknown index")?);
            }
            0x6a => {
                let idx = u32::from_le_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]) as usize;
                i += 4;
                stack.push(memo.get(&idx).cloned().ok_or("LONG_BINGET with unknown index")?);
            }
            0x63 => {
                // GLOBAL: two newline terminated strings
                let (module, next) = read_line(data, i)?;
                let (name, next2) = read_line(data, next)?;
                i = next2;
                stack.push(Val::Global(format!("{module} {name}")));
            }
            0x93 => {
                // STACK_GLOBAL
                let name = pop(&mut stack)?;
                let module = pop(&mut stack)?;
                stack.push(Val::Global(format!("{} {}", as_str(&module)?, as_str(&name)?)));
            }
            0x52 => {
                // REDUCE
                let args = pop(&mut stack)?;
                let func = pop(&mut stack)?;
                stack.push(apply_reduce(&func, &args)?);
            }
            0x62 => {
                // BUILD: the state is not needed for us
                let _state = pop(&mut stack)?;
                let obj = pop(&mut stack)?;
                stack.push(obj);
            }
            0x81 => {
                // NEWOBJ
                let _args = pop(&mut stack)?;
                let cls = pop(&mut stack)?;
                stack.push(Val::Other(format!("newobj {}", as_str(&cls)?)));
            }
            0x51 => {
                // BINPERSID
                let id = pop(&mut stack)?;
                stack.push(into_storage(&id)?);
            }
            other => {
                return Err(format!("unsupported pickle opcode 0x{other:02x} at byte {}", i - 1));
            }
        }
    }
}

fn read_line(data: &[u8], from: usize) -> Result<(String, usize), String> {
    let mut i = from;
    while i < data.len() && data[i] != b'\n' {
        i += 1;
    }
    if i >= data.len() {
        return Err("unterminated line in pickle".into());
    }
    let s = std::str::from_utf8(&data[from..i]).map_err(|_| "bad utf8 in pickle".to_string())?;
    Ok((s.to_string(), i + 1))
}

fn as_str(v: &Val) -> Result<&str, String> {
    match v {
        Val::Str(s) => Ok(s),
        Val::Global(s) => Ok(s),
        _ => Err("expected a string in pickle".into()),
    }
}

fn into_storage(id: &Val) -> Result<Val, String> {
    let parts = match id {
        Val::Tuple(t) => t,
        _ => return Err("persistent id is not a tuple".into()),
    };
    if parts.len() < 5 {
        return Err("unexpected persistent id shape".into());
    }
    if as_str(&parts[0]).unwrap_or("") != "storage" {
        return Err("persistent id is not a storage".into());
    }
    let dtype = match &parts[1] {
        Val::Global(g) => g.clone(),
        _ => return Err("storage dtype is not a global".into()),
    };
    let key = as_str(&parts[2])?.to_string();
    let numel = match &parts[4] {
        Val::Int(n) => *n,
        _ => return Err("storage numel is not an int".into()),
    };
    Ok(Val::Storage { dtype, key, numel })
}

fn apply_reduce(func: &Val, args: &Val) -> Result<Val, String> {
    let name = match func {
        Val::Global(g) => g.as_str(),
        _ => return Ok(Val::Other("reduce".into())),
    };
    let args = match args {
        Val::Tuple(t) => t,
        _ => return Ok(Val::Other(format!("reduce {name}"))),
    };
    if name.ends_with("_rebuild_tensor_v2") || name.ends_with("_rebuild_tensor") {
        if args.len() < 4 {
            return Err("tensor rebuild with too few arguments".into());
        }
        let (dtype, key, storage_numel) = match &args[0] {
            Val::Storage { dtype, key, numel } => (dtype.clone(), key.clone(), *numel),
            _ => return Err("tensor rebuild without storage".into()),
        };
        let offset = match &args[1] {
            Val::Int(n) => *n,
            _ => return Err("tensor offset is not an int".into()),
        };
        let size = int_list(&args[2])?;
        let stride = int_list(&args[3])?;
        return Ok(Val::Tensor(Tensor {
            dtype,
            key,
            storage_numel,
            offset,
            size,
            stride,
        }));
    }
    if name.ends_with("_rebuild_parameter") {
        if let Some(first) = args.first() {
            return Ok(first.clone());
        }
        return Err("_rebuild_parameter without arguments".into());
    }
    if name.ends_with("OrderedDict") || name.ends_with(" dict") {
        return Ok(Val::Dict(Vec::new()));
    }
    if name.ends_with(" set") || name.ends_with("frozenset") {
        return Ok(Val::Set(Vec::new()));
    }
    Ok(Val::Other(format!("reduce {name}")))
}

fn int_list(v: &Val) -> Result<Vec<i64>, String> {
    match v {
        Val::Tuple(t) => t
            .iter()
            .map(|x| match x {
                Val::Int(n) => Ok(*n),
                _ => Err("expected int in shape/stride".to_string()),
            })
            .collect(),
        _ => Err("expected tuple in shape/stride".into()),
    }
}
