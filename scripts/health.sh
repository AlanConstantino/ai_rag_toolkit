#!/bin/bash
# ==============================================================================
# health.sh - Check system health and configuration
# ==============================================================================
#
# WHAT THIS DOES:
#   Performs health checks on the RAG system to verify everything is
#   configured correctly and operational. Checks include:
#   - Database connectivity
#   - API endpoint availability (if configured)
#   - Configuration validation
#
# USAGE:
#   ./scripts/health.sh
#
# EXIT CODES:
#   0 = System is healthy
#   1 = One or more health checks failed
#
# EXAMPLE OUTPUT:
#   System Health Report
#   ====================
#   Database: OK
#   Vector API: OK (or SKIP if not configured)
#   Chat API: OK (or SKIP if not configured)
#   Overall: HEALTHY
#
# WHEN TO USE:
#   - After initial setup to verify configuration
#   - When queries are failing unexpectedly
#   - Before running a large crawl job
#   - As part of deployment/CI checks
#
# REQUIREMENTS:
#   - Python 3.6.5+
#
# ==============================================================================

set -e

# Change to project root directory
cd "$(dirname "$0")/.."

echo "========================================"
echo "  RAG System - Health Check"
echo "========================================"
echo ""

python -m rag_system.main health

# The Python script exits with code 1 if unhealthy
