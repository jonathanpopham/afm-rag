---
name: afm-doctrine-research
description: Research U.S. Army Field Manuals through the local afm-rag MCP server. Use when answering questions on tactics, land navigation, first aid, equipment, or military doctrine from local manuals.
---

# AFM Doctrine Research Workflow

Use the `afm-rag` MCP server tools for all manual lookups.
Do not answer doctrine questions from memory without verifying excerpts in the local index.

## Workflow

1. Call `index_stats` to verify indexed manuals, categories, and `active` versus `superseded` counts.
2. Call `search_manuals` with `status="active"` first when answering current operational doctrine questions.
3. Omit `status` or query `status="superseded"` when researching historical editions or declassified studies.
4. Filter searches with `category` or `subcategory` when `index_stats` shows multiple manual domains.
5. Call `read_manual_section` using the returned `doc_id` when a matched section spans multiple chunks.
6. Call `list_manuals` to locate exact publication titles when keyword searches miss a manual.
7. Run follow-up searches with synonyms or related doctrinal terms before concluding coverage gaps.

## Output Rules

- Cite every factual claim using `[Title (status, edition_date) — Section (doc_id)]`.
- Separate verified manual facts from operational gaps or missing sections.
- Report exact missing topics when the local index lacks relevant excerpts.
- Write summaries in ASD-STE100 Simplified Technical English when requested by the user.
