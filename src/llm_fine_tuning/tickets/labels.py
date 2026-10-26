"""
Author: L. Saetta
Date last modified: 2026-10-26
License: MIT
Description: Allowed ticket-classification labels and output keys, as defined by the classification guide.
"""

CATEGORIES = ("APP-01", "AUTH-01", "DB-01", "DB-02", "DEV-01", "MAIL-01", "MOB-01", "NET-01", "SEC-01", "SRV-01")
SEVERITIES = ("P1", "P2", "P3", "P4")
OUTPUT_KEYS = frozenset({"category", "severity", "summary"})
