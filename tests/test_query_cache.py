"""Tests for query result caching."""

import time
import unittest

from rag_system.utils import LRUCache


class TestLRUCacheBasic(unittest.TestCase):
    """Test basic LRU cache operations."""

    def test_put_and_get(self):
        """put and get should store and retrieve values."""
        cache = LRUCache(max_size=10, ttl_seconds=60)
        cache.put('key1', 'value1')

        result = cache.get('key1')

        self.assertEqual(result, 'value1')

    def test_get_nonexistent_returns_none(self):
        """get should return None for nonexistent keys."""
        cache = LRUCache(max_size=10, ttl_seconds=60)

        result = cache.get('nonexistent')

        self.assertIsNone(result)

    def test_delete_removes_entry(self):
        """delete should remove an entry from the cache."""
        cache = LRUCache(max_size=10, ttl_seconds=60)
        cache.put('key1', 'value1')

        result = cache.delete('key1')

        self.assertTrue(result)
        self.assertIsNone(cache.get('key1'))

    def test_delete_nonexistent_returns_false(self):
        """delete should return False for nonexistent keys."""
        cache = LRUCache(max_size=10, ttl_seconds=60)

        result = cache.delete('nonexistent')

        self.assertFalse(result)

    def test_clear_removes_all_entries(self):
        """clear should remove all entries from the cache."""
        cache = LRUCache(max_size=10, ttl_seconds=60)
        cache.put('key1', 'value1')
        cache.put('key2', 'value2')

        cache.clear()

        self.assertIsNone(cache.get('key1'))
        self.assertIsNone(cache.get('key2'))

    def test_update_existing_key(self):
        """put should update existing key with new value."""
        cache = LRUCache(max_size=10, ttl_seconds=60)
        cache.put('key1', 'value1')
        cache.put('key1', 'value2')

        result = cache.get('key1')

        self.assertEqual(result, 'value2')


class TestLRUCacheEviction(unittest.TestCase):
    """Test LRU cache eviction behavior."""

    def test_evicts_lru_when_full(self):
        """Cache should evict least recently used entry when full."""
        cache = LRUCache(max_size=3, ttl_seconds=60)
        cache.put('key1', 'value1')
        cache.put('key2', 'value2')
        cache.put('key3', 'value3')

        # Adding a fourth item should evict key1 (least recently used)
        cache.put('key4', 'value4')

        self.assertIsNone(cache.get('key1'))
        self.assertIsNotNone(cache.get('key2'))
        self.assertIsNotNone(cache.get('key3'))
        self.assertIsNotNone(cache.get('key4'))

    def test_access_updates_lru_order(self):
        """Accessing an entry should update its LRU position."""
        cache = LRUCache(max_size=3, ttl_seconds=60)
        cache.put('key1', 'value1')
        cache.put('key2', 'value2')
        cache.put('key3', 'value3')

        # Access key1 to make it recently used
        cache.get('key1')

        # Adding key4 should now evict key2 (oldest unaccessed)
        cache.put('key4', 'value4')

        self.assertIsNotNone(cache.get('key1'))
        self.assertIsNone(cache.get('key2'))
        self.assertIsNotNone(cache.get('key3'))
        self.assertIsNotNone(cache.get('key4'))

    def test_eviction_count_in_stats(self):
        """Evictions should be counted in statistics."""
        cache = LRUCache(max_size=2, ttl_seconds=60)
        cache.put('key1', 'value1')
        cache.put('key2', 'value2')
        cache.put('key3', 'value3')  # Evicts key1

        stats = cache.get_stats()

        self.assertEqual(stats['evictions'], 1)


class TestLRUCacheTTL(unittest.TestCase):
    """Test LRU cache TTL (time-to-live) behavior."""

    def test_expired_entry_returns_none(self):
        """Expired entries should return None."""
        cache = LRUCache(max_size=10, ttl_seconds=1)
        cache.put('key1', 'value1')

        # Wait for expiration
        time.sleep(1.1)

        result = cache.get('key1')

        self.assertIsNone(result)

    def test_unexpired_entry_returns_value(self):
        """Unexpired entries should return their value."""
        cache = LRUCache(max_size=10, ttl_seconds=60)
        cache.put('key1', 'value1')

        result = cache.get('key1')

        self.assertEqual(result, 'value1')

    def test_cleanup_expired_removes_old_entries(self):
        """cleanup_expired should remove expired entries."""
        cache = LRUCache(max_size=10, ttl_seconds=1)
        cache.put('key1', 'value1')
        cache.put('key2', 'value2')

        # Wait for expiration
        time.sleep(1.1)

        removed = cache.cleanup_expired()

        self.assertEqual(removed, 2)

    def test_zero_ttl_disables_expiration(self):
        """TTL of 0 should disable expiration."""
        cache = LRUCache(max_size=10, ttl_seconds=0)
        cache.put('key1', 'value1')

        # Even with a pause, entry should not expire
        time.sleep(0.1)

        result = cache.get('key1')

        self.assertEqual(result, 'value1')


class TestLRUCacheStats(unittest.TestCase):
    """Test LRU cache statistics."""

    def test_hit_count(self):
        """Stats should track cache hits."""
        cache = LRUCache(max_size=10, ttl_seconds=60)
        cache.put('key1', 'value1')
        cache.get('key1')
        cache.get('key1')

        stats = cache.get_stats()

        self.assertEqual(stats['hits'], 2)

    def test_miss_count(self):
        """Stats should track cache misses."""
        cache = LRUCache(max_size=10, ttl_seconds=60)
        cache.get('nonexistent')
        cache.get('also_nonexistent')

        stats = cache.get_stats()

        self.assertEqual(stats['misses'], 2)

    def test_hit_rate_calculation(self):
        """Stats should calculate hit rate correctly."""
        cache = LRUCache(max_size=10, ttl_seconds=60)
        cache.put('key1', 'value1')
        cache.get('key1')  # Hit
        cache.get('key1')  # Hit
        cache.get('nonexistent')  # Miss

        stats = cache.get_stats()

        # 2 hits / 3 total = 0.6667
        self.assertAlmostEqual(stats['hit_rate'], 0.6667, places=3)

    def test_size_tracking(self):
        """Stats should track current size."""
        cache = LRUCache(max_size=10, ttl_seconds=60)
        cache.put('key1', 'value1')
        cache.put('key2', 'value2')

        stats = cache.get_stats()

        self.assertEqual(stats['size'], 2)
        self.assertEqual(stats['max_size'], 10)

    def test_reset_stats(self):
        """reset_stats should clear hit/miss/eviction counts."""
        cache = LRUCache(max_size=10, ttl_seconds=60)
        cache.put('key1', 'value1')
        cache.get('key1')
        cache.get('nonexistent')

        cache.reset_stats()
        stats = cache.get_stats()

        self.assertEqual(stats['hits'], 0)
        self.assertEqual(stats['misses'], 0)
        self.assertEqual(stats['evictions'], 0)
        # Size should still be tracked
        self.assertEqual(stats['size'], 1)


class TestLRUCacheThreadSafety(unittest.TestCase):
    """Test LRU cache thread safety."""

    def test_concurrent_put_and_get(self):
        """Cache should handle concurrent access safely."""
        import threading

        cache = LRUCache(max_size=100, ttl_seconds=60)
        errors = []

        def put_values(start, end):
            try:
                for i in range(start, end):
                    cache.put(f'key{i}', f'value{i}')
            except Exception as e:
                errors.append(e)

        def get_values(start, end):
            try:
                for i in range(start, end):
                    cache.get(f'key{i}')
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=put_values, args=(0, 50)),
            threading.Thread(target=put_values, args=(50, 100)),
            threading.Thread(target=get_values, args=(0, 100)),
            threading.Thread(target=get_values, args=(0, 100)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)


class TestLRUCacheComplexValues(unittest.TestCase):
    """Test LRU cache with complex value types."""

    def test_dict_values(self):
        """Cache should store and retrieve dict values."""
        cache = LRUCache(max_size=10, ttl_seconds=60)
        value = {'answer': 'test', 'chunks': [1, 2, 3], 'confidence': 0.9}
        cache.put('key1', value)

        result = cache.get('key1')

        self.assertEqual(result, value)

    def test_list_values(self):
        """Cache should store and retrieve list values."""
        cache = LRUCache(max_size=10, ttl_seconds=60)
        value = [1, 2, 3, 'test', {'nested': True}]
        cache.put('key1', value)

        result = cache.get('key1')

        self.assertEqual(result, value)


if __name__ == '__main__':
    unittest.main()
