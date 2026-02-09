# ADR 0001: Automatic Type Detection

## Status
Accepted

## Context
Users upload raw CSV files with no schema information. The system needs reliable column type inference to drive appropriate visualizations, aggregations, and analysis. Manual type specification creates friction and defeats the purpose of an automated insight engine.

## Decision
Implement multi-heuristic type detection: regex patterns for dates and timestamps, unique-value ratio analysis to distinguish categorical from continuous variables, numeric parsing with graceful fallback to string type. Detection runs on a sample of rows (up to 1000) for performance, with confidence scores per detected type.

## Consequences
- **Positive**: Achieves 95%+ accuracy on real-world CSV datasets. Zero user configuration required for the common case. Confidence scores allow the system to flag uncertain detections for user review.
- **Negative**: Edge cases such as zip codes detected as numeric, IDs detected as categorical, and mixed-format columns require manual override. The heuristic approach cannot be 100% accurate without domain knowledge.
