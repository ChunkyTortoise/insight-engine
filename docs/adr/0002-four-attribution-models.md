# ADR 0002: Four Attribution Models

## Status
Accepted

## Context
Marketing attribution is inherently subjective. Different attribution models favor different channels, and no single model captures the full picture. Stakeholders across an organization often disagree on which channels drive the most value. Offering only one model would embed a bias into the platform.

## Decision
Implement four attribution models: first-touch, last-touch, linear, and time-decay. All four run on the same conversion data and are presented in a side-by-side comparison view. Users can select their preferred model for reporting while seeing how attribution shifts across models.

## Consequences
- **Positive**: Provides a balanced, multi-perspective view of channel value. Users can choose the model that best fits their business model and sales cycle. Side-by-side comparison educates users about attribution nuances and reduces overreliance on any single model.
- **Negative**: Increases UI complexity with four parallel views. Users unfamiliar with attribution theory may not understand the differences between models. Maintaining four model implementations increases testing surface area.
