---
title: CSV Export Failure After Version 3.2 Upgrade
document_type: known_issue
product: InsightFlow
version: "3.2"
category: reporting
updated_at: "2026-06-18"
authority_score: 0.90
reviewed: true
---

# Known Issue

Some customers upgrading from InsightFlow 3.1 to 3.2 receive error EXP-3204
during CSV export.

The issue is caused by cached export settings that reference the deprecated
report schema.

The supported resolution is to clear the report configuration cache and
recreate the export settings.