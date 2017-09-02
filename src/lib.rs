//! In-memory fixed-size cache for arbitrary types.
//!
//! Primarily intended for content-addressable storage, where the key uniquely
//! identifies a value.
#![deny(missing_docs)]
extern crate parking_lot;
extern crate seahash;
extern crate stable_heap;

use std::collections::BTreeMap;
use std::ops::Deref;
use std::time::{Duration, Instant};
use std::marker::PhantomData;
use std::hash::{Hash, Hasher};
use std::mem::{self, ManuallyDrop};
use std::{fmt, ptr};
use std::any::TypeId;

use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use seahash::SeaHasher;

const ALIGNMENT: usize = 8;
const TRIES: usize = 8;

fn wrap_drop<V>(ptr: *const ManuallyDrop<u8>) {
    unsafe {
        let ptr: &mut ManuallyDrop<V> = mem::transmute(ptr);
        ManuallyDrop::drop(ptr);
    }
}

// We use `repr(C)` here, since rust does not currently guarantee field order,
// and we must be able to read the non-val fields in a type-erased way
#[repr(C)]
struct Entry<K, V> {
    key: K,
    lock: RwLock<()>,
    atime: RwLock<Instant>,
    destructor: fn(*const ManuallyDrop<u8>),
    type_id: TypeId,
    // val is of arbitrary size, and has to be on the end
    // to access the other fields with V erased
    val: ManuallyDrop<V>,
}

impl<K, V: 'static> Entry<K, V> {
    fn new(key: K, val: V) -> Self {
        let val = ManuallyDrop::new(val);
        Entry {
            key,
            val,
            destructor: wrap_drop::<V>,
            atime: RwLock::new(Instant::now()),
            lock: RwLock::new(()),
            type_id: TypeId::of::<V>(),
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

struct Page<K> {
    allocations: RwLock<BTreeMap<usize, usize>>,
    size: usize,
    slab: *mut u8,
    _marker: PhantomData<K>,
}

/// A thread-safe concurrent cache from key `K` to arbitrarily typed values
pub struct Cache<K> {
    pages: Vec<Page<K>>,
}

unsafe impl<K> Send for Cache<K> {}
unsafe impl<K> Sync for Cache<K> {}

impl<K: Hash + Eq> Cache<K> {
    /// Create a new cache setting page size and number of pages.
    ///
    /// `page_size` determines the maximum size of values that can be
    /// stored in the cache.
    ///
    /// `num pages` determines how many slabs of memory of this size should be
    /// allocated.
    ///
    /// Each page has its own read-write lock, so the more pages you have,
    /// the less likely you are to have lock contention.
    pub fn new(num_pages: usize, page_size: usize) -> Self {
        assert!(num_pages > 0, "Must have at least one page");
        let mut pages = Vec::with_capacity(num_pages);
        for _ in 0..num_pages {
            pages.push(Page::new(page_size))
        }
        Cache { pages }
    }

    fn hash(key: &K) -> u64 {
        let mut hasher = SeaHasher::new();
        key.hash(&mut hasher);
        hasher.finish()
    }

    /// Insert a value `V` into cache with key `K`, returns a `Reference` to
    /// the newly stored or spilled value
    ///
    /// *NOTE* If you insert different values under the same key, you will only
    /// get one of them out again. Updates are not possible.
    pub fn insert<V: 'static>(
        &self,
        key: K,
        val: V,
    ) -> Reference<V> {
        let hash = Self::hash(&key);
        let page = hash as usize % self.pages.len();
        self.pages[page].insert(hash, key, val)
    }

    /// Is this value in the cache?
    pub fn get<'a, V: 'static>(
        &'a self,
        key: &K,
    ) -> Option<Reference<'a, V>> {
        let hash = Self::hash(&key);
        let page = hash as usize % self.pages.len();
        self.pages[page].get(hash, key)
    }
}

/// A reference type to a value in the cache, either carrying the value
/// itself (if an insert failed), or a readlock into the memory slab.
pub enum Reference<'a, V> {
    /// Value could not be put on the cache, and is returned as-is
    Spilled(V),
    /// Value resides in cache and is read-locked.
    Cached {
        /// The readguard from a lock on the heap
        guard: RwLockReadGuard<'a, ()>,
        /// A pointer to a value on the heap
        ptr: *const ManuallyDrop<V>,
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

    fn offset<V: 'static>(&self, hash: u64) -> usize {
        let size = Entry::<K, V>::size();
        let offset = (hash as usize % (self.size / ALIGNMENT)) * ALIGNMENT;
        if offset + size > self.size {
            // wraparound
            0
        } else {
            offset
        }
    }

    pub fn insert<V: 'static>(
        &self,
        hash: u64,
        key: K,
        val: V,
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
                    ptr::write(ptr, entry);
                    let vptr = &(*ptr).val;
                    let guard = (*ptr).lock.read();
                    return Reference::Cached { ptr: vptr, guard };
                }
            } else {
                match self.evict(orig_offset, size, &mut allocations) {
                    Some(evict) => {
                        for &(offset, ref destructor) in &evict {
                            // run destructor and deallocate
                            unsafe {
                                let entry: &Entry<K, u8> = mem::transmute(self.slab.offset(offset as isize));
                                destructor(&entry.val);
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
    ) -> Option<Vec<(usize, &fn(*const ManuallyDrop<u8>))>> {
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
                            let destructor;
                            elapsed_times.push(unsafe {
                                let ptr =
                                    self.slab.offset(*entry_offset as isize);
                                let entry: &Entry<K, ()> = mem::transmute(ptr);
                                destructor = &entry.destructor;
                                entry.atime.read().elapsed()
                            });
                            match last_end {
                                Some(n) => {
                                    // empty space after last entry
                                    full_size += entry_offset - n;
                                }
                                None => (),
                            }

                            elements.push((*entry_offset, destructor));
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

                // make sure the type is right
                if entry.type_id == TypeId::of::<V>() && &entry.key == key {
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

    #[test]
    fn wrong_type() {
        let cache = Cache::new(1, 4096);

        cache.insert(0usize, 0usize);

        assert!(cache.get::<u32>(&0).is_none());
        assert!(cache.get::<usize>(&0).is_some());
    }
}
