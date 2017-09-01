extern crate parking_lot;
extern crate seahash;
extern crate stable_heap;

use std::collections::BTreeMap;
use std::ops::Deref;
use std::time::{Duration, Instant};
use std::marker::PhantomData;
use std::hash::{Hash, Hasher};
use std::{fmt, mem, ptr};
use std::any::TypeId;

use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use seahash::SeaHasher;

const ALIGNMENT: usize = 8;
const TRIES: usize = 8;

// We use `repr(C)` here, since rust does not currently guarantee field order,
// and we must be able to read the non-val fields in a type-erased way
#[repr(C)]
struct Entry<K, V> {
    key: K,
    lock: RwLock<()>,
    atime: RwLock<Instant>,
    type_id: TypeId,
    // val is of arbitrary size, and has to be on the end
    // to access the other fields with V erased
    val: V,
}

impl<K, V: 'static> Entry<K, V> {
    fn new(key: K, val: V) -> Self {
        Entry {
            key,
            val,
            type_id: TypeId::of::<V>(),
            atime: RwLock::new(Instant::now()),
            lock: RwLock::new(()),
        }
    }

    // size of entry rounded up to nearest alignment
    fn size() -> usize {
        let size = mem::size_of::<Self>();
        let overlap = size % ALIGNMENT;
        let mut size = size / ALIGNMENT;
        if overlap != 0 {
            size += 1;
        }
        size * ALIGNMENT
    }
}

pub struct Page<K> {
    allocations: RwLock<BTreeMap<usize, usize>>,
    size: usize,
    slab: *mut u8,
    _marker: PhantomData<K>,
}

type Destructors = BTreeMap<TypeId, Box<Fn(*const u8)>>;

pub struct Cache<K> {
    pages: Vec<Page<K>>,
    destructors: RwLock<Destructors>,
}

unsafe impl<K> Send for Cache<K> {}
unsafe impl<K> Sync for Cache<K> {}

impl<K: Hash + Eq> Cache<K> {
    pub fn new(num_pages: usize, page_size: usize) -> Self {
        assert!(num_pages > 0, "Must have at least one page");
        let mut pages = Vec::with_capacity(num_pages);
        for _ in 0..num_pages {
            pages.push(Page::new(page_size))
        }
        Cache {
            pages,
            destructors: RwLock::new(BTreeMap::new()),
        }
    }

    fn hash(key: &K) -> u64 {
        let mut hasher = SeaHasher::new();
        key.hash(&mut hasher);
        hasher.finish()
    }

    pub fn insert<V: Sized + 'static>(&self, key: K, val: V) -> Reference<V> {
        let hash = Self::hash(&key);
        let page = hash as usize % self.pages.len();
        let type_id = TypeId::of::<V>();
        // cache a destructor for this value if not already cached
        if self.destructors.read().get(&type_id).is_none() {
            println!("inserting type {:?}", type_id);
            self.destructors.write().insert(
                type_id,
                Box::new(|addr| unsafe {
                    let ptr: *const V = mem::transmute(addr);
                    println!("read ptr {:?}", ptr);
                    let _ = ptr::read(ptr);
                }),
            );
        }
        self.pages[page].insert(hash, key, val, &*self.destructors.read())
    }

    pub fn get<'a, V: 'static>(&'a self, key: &K) -> Option<Reference<'a, V>> {
        let hash = Self::hash(&key);
        let page = hash as usize % self.pages.len();
        self.pages[page].get(hash, key)
    }
}

pub enum Reference<'a, V> {
    Spilled(V),
    Cached {
        guard: RwLockReadGuard<'a, ()>,
        ptr: *const V,
    },
}

impl<'a, V> fmt::Debug for Reference<'a, V>
where
    V: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Reference::Spilled(ref v) => write!(f, "Spilled({:?})", v),
            Reference::Cached { ptr, .. } => {
                write!(f, "Cached({:?})", unsafe { &*ptr })
            }
        }
    }
}

impl<'a, V> Deref for Reference<'a, V> {
    type Target = V;
    fn deref(&self) -> &V {
        match *self {
            Reference::Spilled(ref v) => v,
            Reference::Cached { ptr, .. } => unsafe { &*ptr },
        }
    }
}

impl<K: Hash + Eq> Page<K> {
    pub fn new(size: usize) -> Self {
        Page {
            allocations: RwLock::new(BTreeMap::new()),
            slab: unsafe {
                stable_heap::allocate(size, mem::align_of::<usize>())
            },
            _marker: PhantomData,
            size,
        }
    }

    fn offset<V: Sized + 'static>(&self, hash: u64) -> usize {
        let size = Entry::<K, V>::size();
        let offset = (hash as usize % (self.size / ALIGNMENT)) * ALIGNMENT;
        if offset + size > self.size {
            // wraparound
            0
        } else {
            offset
        }
    }

    pub fn insert<V: Sized + 'static>(
        &self,
        hash: u64,
        key: K,
        val: V,
        destructors: &Destructors,
    ) -> Reference<V> {
        let size = Entry::<K, V>::size();
        let orig_offset = self.offset::<V>(hash);
        let mut offset = orig_offset;
        let mut allocations = self.allocations.write();
        let mut found = None;
        {
            let mut tries = TRIES + 1;
            loop {
                tries -= 1;
                if tries == 0 {
                    break;
                }

                // enough room until end?
                if offset + size > self.size {
                    // wraparound
                    offset = 0;
                }

                // check the area before this offset
                match allocations.range(..offset).next_back() {
                    None => (),
                    Some(before) => if offset >= before.0 + before.1 {
                        // before with enough room
                    } else {
                        // before with not enough room;
                        offset = before.0 + before.1;
                        continue;
                    },
                }

                // check the area after this offset
                match allocations.range(offset..).next() {
                    None => {
                        found = Some(offset);
                        break;
                    }
                    Some(after) => if offset + size < *after.0 {
                        // after with enough room"
                        found = Some(offset);
                        break;
                    } else {
                        // after with not enough room
                        offset = after.0 + after.1;
                        continue;
                    },
                }
            }
        }
        loop {
            if let Some(found) = found {
                let entry = Entry::new(key, val);
                debug_assert!(found + size <= self.size);
                allocations.insert(found, size);
                unsafe {
                    let ptr = self.slab.offset(found as isize);
                    let ptr: *mut Entry<K, V> = mem::transmute(ptr);
                    println!("write ptr {:?}", ptr);
                    ptr::write(ptr, entry);
                    let vptr = &(*ptr).val;
                    let guard = (*ptr).lock.read();

                    return Reference::Cached { ptr: vptr, guard };
                }
            } else {
                match self.evict(orig_offset, size, &mut allocations) {
                    Some(evict) => {
                        for &(offset, type_id) in &evict {
                            // run destructor and deallocate
                            unsafe {
                                println!("destroying {:?}", type_id);
                                let ptr = self.slab.offset(offset as isize);
                                destructors
                                    .get(&type_id)
                                    .expect("No destructor for type")(ptr);
                            }
                            allocations.remove(&offset);
                        }
                        found = Some(evict.first().expect("len > 1").0);
                    }
                    None => {
                        return Reference::Spilled(val);
                    }
                }
            }
        }
    }

    fn evict(
        &self,
        from: usize,
        size: usize,
        allocations: &mut RwLockWriteGuard<BTreeMap<usize, usize>>,
    ) -> Option<Vec<(usize, TypeId)>> {
        let mut ranges = vec![];
        let mut adjacent = vec![];
        {
            // wraparound
            let check = allocations.range(from..).chain(allocations.iter());
            let mut last_offset = 0;
            for (offset, size) in check.take(TRIES) {
                unsafe {
                    let ptr = self.slab.offset(*offset as isize);
                    let entry: &mut Entry<K, ()> = mem::transmute(ptr);

                    match entry.lock.try_write() {
                        Some(writeguard) => {
                            if last_offset > *offset {
                                // we wrapped
                                if adjacent.len() > 0 {
                                    ranges.push(
                                        mem::replace(&mut adjacent, vec![]),
                                    )
                                }
                            }
                            adjacent.push((offset, size, writeguard));
                        }
                        None => if adjacent.len() > 0 {
                            ranges.push(mem::replace(&mut adjacent, vec![]))
                        },
                    }
                }
                last_offset = *offset;
            }
        }
        if adjacent.len() > 0 {
            ranges.push(mem::replace(&mut adjacent, vec![]))
        }

        let mut candidates = BTreeMap::new();

        for range in ranges {
            // all the allocations in a range are adjacent and writelocked
            'range: for i in 0..range.len() {
                let mut check = i;
                let mut elapsed_times = vec![];
                let mut elements = vec![];
                let mut full_size = 0;
                let mut last_end = None;
                // select n slots to evict to make enough room for `size`
                loop {
                    match range.get(check) {
                        None => break 'range,
                        Some(&(entry_offset, entry_size, _)) => {
                            let type_id;
                            elapsed_times.push(unsafe {
                                let ptr =
                                    self.slab.offset(*entry_offset as isize);
                                let entry: &Entry<K, ()> = mem::transmute(ptr);
                                type_id = entry.type_id;
                                entry.atime.read().elapsed()
                            });
                            match last_end {
                                Some(n) => {
                                    // empty space after last entry
                                    full_size += entry_offset - n;
                                }
                                None => (),
                            }

                            elements.push((*entry_offset, type_id));
                            full_size += *entry_size;
                            last_end = Some(entry_offset + entry_size);

                            if full_size >= size {
                                let mut elapsed = elapsed_times.iter().fold(
                                    Duration::from_millis(0),
                                    |a, b| a + *b,
                                );
                                elapsed /= elapsed_times.len() as u32;

                                candidates.insert(elapsed, elements);
                                break;
                            }
                        }
                    }
                    check += 1;
                }
            }
        }

        candidates
            .iter_mut()
            // pick out oldest value
            .last()
            .map(|(_, val)| mem::replace(val, vec![]))
    }

    pub fn get<'a, V: 'static>(
        &'a self,
        hash: u64,
        key: &K,
    ) -> Option<Reference<'a, V>> {
        let offset = self.offset::<V>(hash);
        let allocations = self.allocations.read();

        let check = allocations.range(offset..).chain(allocations.iter());

        for (offset, _) in check.take(TRIES) {
            unsafe {
                let ptr = self.slab.offset(*offset as isize);
                let entry: &Entry<K, V> = mem::transmute(ptr);

                if &entry.key == key {
                    *entry.atime.write() = Instant::now();
                    return Some(Reference::Cached {
                        guard: entry.lock.read(),
                        ptr: &entry.val,
                    });
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;
    use std::thread;

    #[test]
    fn usizes() {
        let cache = Cache::new(1, 4096);

        let n: usize = 10_000;

        for i in 0..n {
            assert_eq!(*cache.insert(i, i), i);
            let gotten = cache.get::<usize>(&i);
            assert_eq!(*gotten.unwrap(), i);
        }
        // 0 should have fallen out by now!
        assert!(cache.get::<usize>(&0).is_none());
    }

    #[test]
    fn keepalive() {
        let cache = Cache::new(1, 4096);

        let n: usize = 10_000;

        for i in 0..n {
            assert_eq!(*cache.insert(i, i), i);
            let gotten = cache.get::<usize>(&i);
            assert_eq!(*gotten.unwrap(), i);
            let _keepalive = cache.get::<usize>(&0);
        }
        // 0 should have been kept alive!
        assert_eq!(*cache.get::<usize>(&0).unwrap(), 0);
    }


    #[test]
    fn destructors() {
        let cache = Cache::new(1, 4096);

        let arc = Arc::new(42usize);

        println!("arc");
        cache.insert(0, arc.clone());
        assert_eq!(Arc::strong_count(&arc), 2);

        let n: usize = 10_000;

        // spam usizes to make arc fall out
        for i in 1..n {
            cache.insert(i, i);
        }
        // arc should have fallen out
        assert!(cache.get::<Arc<usize>>(&0).is_none());
        // and had its destructor run
        assert_eq!(Arc::strong_count(&arc), 1);
    }

    #[test]
    fn larger() {
        let cache = Cache::new(1, 4096);

        let n: usize = 1_000;

        for i in 0..n {
            let block = [i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i];
            assert_eq!(*cache.insert(i, block.clone()), block.clone());
            let gotten = cache.get::<[usize; 16]>(&i);
            assert_eq!(*gotten.unwrap(), block);
        }
    }

    #[test]
    fn multithreading() {
        let cache = Arc::new(Cache::new(32, 4096));

        let n: usize = 1000;

        let mut handles = vec![];

        for i in 0..n {
            let cache = cache.clone();
            handles.push(thread::spawn(move || {
                let block = [i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i];
                assert_eq!(*cache.insert(i, block.clone()), block.clone());
            }))
        }

        for handle in handles {
            handle.join().unwrap()
        }
    }

    #[test]
    fn lots() {
        let cache = Cache::new(32, 4096);

        let n: usize = 100_000;

        for i in 0..n {
            let block = [i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i];
            assert_eq!(*cache.insert(i, block.clone()), block.clone());
            let gotten = cache.get::<[usize; 16]>(&i);
            assert_eq!(*gotten.unwrap(), block);
        }
    }

    #[test]
    fn evict_multiple() {
        let cache = Cache::new(1, 4096);

        // first fill cache with small values
        let n: usize = 1000;
        for i in 0..n {
            assert_eq!(*cache.insert(i, i), i);
            let gotten = cache.get::<usize>(&i);
            assert_eq!(*gotten.unwrap(), i);
        }

        // then evict them with larger
        let n2: usize = 1_000;
        for i in 0..n2 {
            let block = [i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i];
            assert_eq!(*cache.insert(i + n, block.clone()), block.clone());
            let gotten = cache.get::<[usize; 16]>(&(i + n));
            assert_eq!(*gotten.unwrap(), block);
        }
        // 0 should have fallen out by now!
        assert!(cache.get::<usize>(&0).is_none());
    }

    #[test]
    fn evict_multiple_keepalive() {
        let cache = Cache::new(1, 4096);

        cache.insert(0usize, 0usize);

        // then evict them with larger
        let n: usize = 1_000;
        for i in 1..n + 1 {
            let block = [i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i];
            assert_eq!(*cache.insert(i, block.clone()), block.clone());
            let gotten = cache.get::<[usize; 16]>(&i);
            assert_eq!(*gotten.unwrap(), block);
            let _keepalive = cache.get::<usize>(&0);
        }
        // 0 should have been kept alive!
        assert_eq!(*cache.get::<usize>(&0).unwrap(), 0);
    }

    #[test]
    fn big_then_small() {
        let cache = Cache::new(1, 4096);

        // first fill cache with large values
        let n: usize = 1_000;
        for i in 0..n {
            let block = [i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i];
            assert_eq!(*cache.insert(i, block.clone()), block.clone());
            let gotten = cache.get::<[usize; 16]>(&i);
            assert_eq!(*gotten.unwrap(), block);
        }

        // then evict them with smaller
        let n2: usize = 1000;
        for i in 0..n2 {
            assert_eq!(*cache.insert(i + n, i), i);
            let gotten = cache.get::<usize>(&(i + n));
            assert_eq!(*gotten.unwrap(), i);
        }
    }
}
