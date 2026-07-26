---
title: InsightFlow Version 3.2 Release Notes
document_type: release_note
product: InsightFlow
version: "3.2"
category: reporting
updated_at: "2026-06-15"
authority_score: 0.93
reviewed: true
---

# Report Export Changes

InsightFlow 3.2 introduced a new report export schema.

Report configurations created in versions earlier than 3.2 may remain cached
using the older schema.

This can produce error EXP-3204 when users attempt to export reports as CSV.

To resolve the issue:

1. Clear the report configuration cache.
2. Recreate the affected export configuration.
3. Retry the CSV export.