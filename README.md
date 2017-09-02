[Documentation](https://krl.github.io/rustdoc/cache/cache/index.html)

# Cache

Fixed-size LRU-Cache capable of caching values of different type.

# Example

Example demonstrating caching two different types in the cache, and having destructors run.

```rust
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
```