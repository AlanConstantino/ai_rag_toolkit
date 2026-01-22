"""Tests for structured logging and metrics collection."""

import json
import logging
import os
import tempfile
import threading
import time
import unittest

from rag_system.utils import (
    JSONFormatter,
    get_log_format,
    get_log_level,
    get_logger,
    log_with_context,
    MetricsCollector,
    get_metrics_collector,
    Timer,
    timed_operation,
)
from rag_system.database import (
    init_db,
    get_connection,
    log_query,
    get_query_logs,
    get_query_analytics,
    migrate_query_log_timing,
    export_query_logs_csv,
)


class TestJSONFormatter(unittest.TestCase):
    """Tests for JSONFormatter class."""

    def test_format_basic_message(self):
        """JSONFormatter should format a basic log message as JSON."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name='test',
            level=logging.INFO,
            pathname='',
            lineno=0,
            msg='Test message',
            args=(),
            exc_info=None
        )

        output = formatter.format(record)
        data = json.loads(output)

        self.assertEqual(data['level'], 'INFO')
        self.assertEqual(data['logger'], 'test')
        self.assertEqual(data['message'], 'Test message')
        self.assertIn('timestamp', data)

    def test_format_with_extra_data(self):
        """JSONFormatter should include extra data when present."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name='test',
            level=logging.INFO,
            pathname='',
            lineno=0,
            msg='Test message',
            args=(),
            exc_info=None
        )
        record.extra_data = {'key': 'value', 'count': 42}

        output = formatter.format(record)
        data = json.loads(output)

        self.assertEqual(data['data']['key'], 'value')
        self.assertEqual(data['data']['count'], 42)

    def test_format_with_exception(self):
        """JSONFormatter should include exception info when present."""
        formatter = JSONFormatter()

        try:
            raise ValueError("Test error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name='test',
            level=logging.ERROR,
            pathname='',
            lineno=0,
            msg='Error occurred',
            args=(),
            exc_info=exc_info
        )

        output = formatter.format(record)
        data = json.loads(output)

        self.assertIn('exception', data)
        self.assertIn('ValueError', data['exception'])


class TestLogConfiguration(unittest.TestCase):
    """Tests for log configuration functions."""

    def setUp(self):
        """Save original environment."""
        self.orig_log_format = os.environ.get('RAG_LOG_FORMAT')
        self.orig_log_level = os.environ.get('RAG_LOG_LEVEL')

    def tearDown(self):
        """Restore original environment."""
        if self.orig_log_format:
            os.environ['RAG_LOG_FORMAT'] = self.orig_log_format
        elif 'RAG_LOG_FORMAT' in os.environ:
            del os.environ['RAG_LOG_FORMAT']

        if self.orig_log_level:
            os.environ['RAG_LOG_LEVEL'] = self.orig_log_level
        elif 'RAG_LOG_LEVEL' in os.environ:
            del os.environ['RAG_LOG_LEVEL']

    def test_get_log_format_default(self):
        """get_log_format should return 'text' by default."""
        if 'RAG_LOG_FORMAT' in os.environ:
            del os.environ['RAG_LOG_FORMAT']

        self.assertEqual(get_log_format(), 'text')

    def test_get_log_format_json(self):
        """get_log_format should return 'json' when configured."""
        os.environ['RAG_LOG_FORMAT'] = 'json'

        self.assertEqual(get_log_format(), 'json')

    def test_get_log_level_default(self):
        """get_log_level should return INFO by default."""
        if 'RAG_LOG_LEVEL' in os.environ:
            del os.environ['RAG_LOG_LEVEL']

        self.assertEqual(get_log_level(), logging.INFO)

    def test_get_log_level_debug(self):
        """get_log_level should return DEBUG when configured."""
        os.environ['RAG_LOG_LEVEL'] = 'DEBUG'

        self.assertEqual(get_log_level(), logging.DEBUG)


class TestMetricsCollector(unittest.TestCase):
    """Tests for MetricsCollector class."""

    def setUp(self):
        """Clear the metrics collector before each test."""
        collector = get_metrics_collector()
        collector.clear()

    def test_singleton_pattern(self):
        """MetricsCollector should be a singleton."""
        collector1 = MetricsCollector()
        collector2 = MetricsCollector()

        self.assertIs(collector1, collector2)

    def test_get_metrics_collector(self):
        """get_metrics_collector should return the singleton."""
        collector1 = get_metrics_collector()
        collector2 = get_metrics_collector()

        self.assertIs(collector1, collector2)

    def test_record_single_metric(self):
        """record should store a single metric value."""
        collector = get_metrics_collector()
        collector.record('test_metric', 1.5)

        metrics = collector.get_all_metrics()
        self.assertEqual(metrics['test_metric'], [1.5])

    def test_record_multiple_metrics(self):
        """record should store multiple metric values."""
        collector = get_metrics_collector()
        collector.record('test_metric', 1.0)
        collector.record('test_metric', 2.0)
        collector.record('test_metric', 3.0)

        metrics = collector.get_all_metrics()
        self.assertEqual(metrics['test_metric'], [1.0, 2.0, 3.0])

    def test_aggregate_metrics(self):
        """get_aggregate_metrics should return statistics."""
        collector = get_metrics_collector()
        collector.record('test_metric', 1.0)
        collector.record('test_metric', 2.0)
        collector.record('test_metric', 3.0)

        aggregates = collector.get_aggregate_metrics()

        self.assertEqual(aggregates['test_metric']['count'], 3)
        self.assertEqual(aggregates['test_metric']['min'], 1.0)
        self.assertEqual(aggregates['test_metric']['max'], 3.0)
        self.assertEqual(aggregates['test_metric']['avg'], 2.0)
        self.assertEqual(aggregates['test_metric']['total'], 6.0)

    def test_query_metrics_tracking(self):
        """start_query/record_query_metric/finish_query should track query metrics."""
        collector = get_metrics_collector()

        collector.start_query()
        collector.record_query_metric('embedding_time', 0.1)
        collector.record_query_metric('search_time', 0.2)

        time.sleep(0.01)  # Small delay
        metrics = collector.finish_query()

        self.assertIn('embedding_time', metrics)
        self.assertIn('search_time', metrics)
        self.assertIn('total_time', metrics)
        self.assertEqual(metrics['embedding_time'], 0.1)
        self.assertEqual(metrics['search_time'], 0.2)
        self.assertGreater(metrics['total_time'], 0)

    def test_clear_metrics(self):
        """clear should remove all metrics."""
        collector = get_metrics_collector()
        collector.record('test_metric', 1.0)

        collector.clear()

        metrics = collector.get_all_metrics()
        self.assertEqual(metrics, {})

    def test_thread_safety(self):
        """MetricsCollector should be thread-safe."""
        collector = get_metrics_collector()
        results = []

        def record_metrics():
            for i in range(100):
                collector.record('thread_metric', float(i))
            results.append(True)

        threads = [threading.Thread(target=record_metrics) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        metrics = collector.get_all_metrics()
        self.assertEqual(len(metrics['thread_metric']), 500)

    def test_to_dict(self):
        """to_dict should return exportable format."""
        collector = get_metrics_collector()
        collector.record('test_metric', 1.0)
        collector.record('test_metric', 2.0)

        data = collector.to_dict()

        self.assertIn('aggregate', data)
        self.assertIn('raw', data)
        self.assertEqual(data['raw']['test_metric'], [1.0, 2.0])


class TestTimedOperation(unittest.TestCase):
    """Tests for timed_operation decorator."""

    def setUp(self):
        """Clear the metrics collector before each test."""
        collector = get_metrics_collector()
        collector.clear()

    def test_timed_operation_records_time(self):
        """timed_operation should record execution time."""
        @timed_operation('test_op')
        def slow_function():
            time.sleep(0.05)
            return 'result'

        result = slow_function()

        self.assertEqual(result, 'result')
        collector = get_metrics_collector()
        metrics = collector.get_all_metrics()
        self.assertIn('test_op', metrics)
        self.assertGreater(metrics['test_op'][0], 0.04)


class TestDatabaseAnalytics(unittest.TestCase):
    """Tests for database analytics functions."""

    def setUp(self):
        """Set up test database."""
        self.temp_file = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.db_path = self.temp_file.name
        self.temp_file.close()
        init_db(self.db_path)

    def tearDown(self):
        """Clean up test database."""
        os.unlink(self.db_path)

    def test_migrate_query_log_timing_new_db(self):
        """migrate_query_log_timing should add timing columns."""
        conn = get_connection(self.db_path)
        try:
            result = migrate_query_log_timing(conn)

            # Check columns exist
            cursor = conn.execute("PRAGMA table_info(query_log)")
            columns = {row['name'] for row in cursor.fetchall()}

            self.assertIn('total_time_ms', columns)
            self.assertIn('embedding_time_ms', columns)
            self.assertIn('metrics_json', columns)
        finally:
            conn.close()

    def test_migrate_query_log_timing_already_migrated(self):
        """migrate_query_log_timing should return False if already migrated."""
        conn = get_connection(self.db_path)
        try:
            # First migration
            result1 = migrate_query_log_timing(conn)
            # Second migration should be no-op
            result2 = migrate_query_log_timing(conn)

            self.assertFalse(result2)
        finally:
            conn.close()

    def test_log_query_with_metrics(self):
        """log_query should store timing metrics."""
        conn = get_connection(self.db_path)
        try:
            migrate_query_log_timing(conn)

            metrics = {
                'total_time': 0.5,
                'embedding_time': 0.1,
                'search_time': 0.2,
                'rerank_time': 0.05,
                'generation_time': 0.15
            }

            log_id = log_query(
                conn, "test query", "factual", ["expanded"], [1, 2, 3],
                0.85, True, metrics=metrics
            )

            # Verify stored data
            cursor = conn.execute("SELECT * FROM query_log WHERE id = ?", (log_id,))
            row = cursor.fetchone()

            self.assertEqual(row['total_time_ms'], 500.0)
            self.assertEqual(row['embedding_time_ms'], 100.0)
            self.assertEqual(row['search_time_ms'], 200.0)
        finally:
            conn.close()

    def test_get_query_analytics_empty(self):
        """get_query_analytics should handle empty database."""
        conn = get_connection(self.db_path)
        try:
            analytics = get_query_analytics(conn)

            self.assertEqual(analytics['total_queries'], 0)
            self.assertEqual(analytics['queries_with_answers'], 0)
        finally:
            conn.close()

    def test_get_query_analytics_with_data(self):
        """get_query_analytics should return correct statistics."""
        conn = get_connection(self.db_path)
        try:
            migrate_query_log_timing(conn)

            # Add some test queries
            log_query(conn, "q1", "factual", [], [], 0.8, True,
                      metrics={'total_time': 0.5})
            log_query(conn, "q2", "howto", [], [], 0.6, True,
                      metrics={'total_time': 0.3})
            log_query(conn, "q3", "factual", [], [], 0.9, False,
                      metrics={'total_time': 0.4})

            analytics = get_query_analytics(conn)

            self.assertEqual(analytics['total_queries'], 3)
            self.assertEqual(analytics['queries_with_answers'], 2)
            self.assertEqual(analytics['query_types']['factual'], 2)
            self.assertEqual(analytics['query_types']['howto'], 1)
        finally:
            conn.close()

    def test_export_query_logs_csv(self):
        """export_query_logs_csv should create valid CSV."""
        conn = get_connection(self.db_path)
        try:
            migrate_query_log_timing(conn)

            # Add some test queries
            log_query(conn, "q1", "factual", [], [], 0.8, True)
            log_query(conn, "q2", "howto", [], [], 0.6, True)

            # Export to temp file
            csv_file = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
            csv_path = csv_file.name
            csv_file.close()

            try:
                count = export_query_logs_csv(conn, csv_path)

                self.assertEqual(count, 2)

                # Verify CSV content
                with open(csv_path, 'r') as f:
                    content = f.read()
                    self.assertIn('query', content)
                    self.assertIn('q1', content)
                    self.assertIn('q2', content)
            finally:
                os.unlink(csv_path)
        finally:
            conn.close()

    def test_export_query_logs_csv_with_limit(self):
        """export_query_logs_csv should respect limit parameter."""
        conn = get_connection(self.db_path)
        try:
            # Add test queries
            for i in range(5):
                log_query(conn, f"q{i}", "factual", [], [], 0.8, True)

            csv_file = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
            csv_path = csv_file.name
            csv_file.close()

            try:
                count = export_query_logs_csv(conn, csv_path, limit=2)

                self.assertEqual(count, 2)
            finally:
                os.unlink(csv_path)
        finally:
            conn.close()


class TestTimer(unittest.TestCase):
    """Tests for Timer context manager."""

    def test_timer_records_elapsed(self):
        """Timer should record elapsed time."""
        with Timer() as t:
            time.sleep(0.05)

        self.assertGreater(t.elapsed, 0.04)
        self.assertLess(t.elapsed, 0.2)

    def test_timer_start_end_times(self):
        """Timer should record start and end times."""
        with Timer() as t:
            time.sleep(0.001)  # Small delay to ensure different times

        self.assertIsNotNone(t.start_time)
        self.assertIsNotNone(t.end_time)
        self.assertGreaterEqual(t.end_time, t.start_time)


if __name__ == '__main__':
    unittest.main()
