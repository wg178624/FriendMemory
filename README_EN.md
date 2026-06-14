# FriendMemory Project

A relational memory system for friend-like AI companions.

> This project explores how an AI can be perceived as "knowing me, understanding me, and sharing a past with me" — and how its memory system should be organized to achieve that.

## What Makes This Different

Traditional memory systems break conversations into factual entries — preferences, names, tasks, or past questions. Friend-like relationships need a more complex memory structure. Certain moments matter not because they contain explicit facts, but because they carry **trust, vulnerability, shared naming, unfulfilled promises, or relationship stage transitions**.

FriendMemory treats this kind of long-term relational memory as a complete system design — it tries to decompose the "friend feeling" into observable, controllable, and verifiable mechanisms, rather than relying only on the model's ad-hoc intimacy tactics.

## Core Capabilities

- **Relationship-perspective memory** — not just user facts, but how the relationship evolves, which interactions changed trust and intimacy
- **Shared experience sedimentation** — recurring events, nicknames, inside jokes, promises, and turning points organized as shared stories
- **Emotional and importance judgment** — distinguish ordinary messages from emotional moments, vulnerable expressions, major commitments, and core identity signals
- **Friend-like recall** — bring up old memories naturally at the right time, continue unfinished topics, remember promises and anniversaries — without spamming or mechanical repetition
- **Relationship stage awareness** — recognize stranger → acquaintance → close → conflict → repair transitions, adapting memory use to current relationship state
- **User control** — users can view, modify, suppress, delete, reset, or export relational memories
- **Safety boundaries** — extra protection for dependency risk, minors, crisis signals, and sensitive information
- **AI participation explainability** — distinguish local rules, external model judgment, and fallback scenarios

## Project Status

**Experimental prototype.** 4 days old, single author. The core idea is solid but implementation is early-stage. Contributions and discussions are welcome.

## Design Principles

This project emphasizes **relational memory over knowledge-base memory**. It's not about storing everything permanently, but about:
- What deserves to be remembered
- What should gradually fade
- What must be confirmed by the user
- What must fully exit the relationship context when deleted

The complete memory lifecycle — generation, retrieval, consolidation, decay, correction, deletion, audit — should all have clear rules.

## Inspiration

This project was partially inspired by the design patterns seen in [Yudustrum](https://github.com/wg178624/yudustrum) (羽渡尘 v5.0), a GraphRAG memory system with brief/full dual storage, cache alignment, and multi-recall retrieval.
